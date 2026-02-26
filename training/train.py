"""Main training script for FastGAN on fractal flame images."""

import copy
import logging
from pathlib import Path

import click
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from model.generator import Generator
from model.discriminator import Discriminator
from model.augment import diff_augment
from model.losses import hinge_loss_d, hinge_loss_g, reconstruction_loss
from data.dataset import FlameDataset
from utils.mps_compat import setup_mps_env, get_device, validate_gradients, mps_empty_cache
from utils.checkpoint import save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    """Update exponential moving average of model weights."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def generate_samples(
    generator: torch.nn.Module,
    fixed_z: torch.Tensor,
    step: int,
    sample_dir: Path,
    writer: SummaryWriter | None,
    grid_size: int = 8,
):
    """Generate and save a grid of sample images."""
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_z)
    generator.train()

    # Denormalize from [-1,1] to [0,1]
    fake = (fake + 1) / 2

    grid = make_grid(fake[:grid_size**2], nrow=grid_size, padding=2)

    # Save image
    save_image(grid, sample_dir / f"step_{step:07d}.png")

    # Log to tensorboard
    if writer is not None:
        writer.add_image("samples", grid, step)


@click.command()
@click.option("--config", "-c", default="configs/default.yaml", help="Config file path.")
@click.option("--resume", "-r", default=None, help="Checkpoint path to resume from.")
def main(config: str, resume: str | None):
    """Train FastGAN on fractal flame images."""
    setup_mps_env()

    # Load config
    with open(config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "mps"))

    # Create output directories
    sample_dir = Path(cfg["sample_dir"])
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(cfg["log_dir"])

    # Dataset
    logger.info(f"Loading dataset from {cfg['dataset_path']}")
    dataset = FlameDataset(cfg["dataset_path"], image_size=cfg["image_size"])
    logger.info(f"Dataset size: {len(dataset)} images")

    num_workers = cfg.get("num_workers", 4)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # MPS doesn't benefit from pin_memory
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    # Models
    G = Generator(latent_dim=cfg["latent_dim"], base_channels=cfg["gen_base_channels"]).to(device)
    D = Discriminator(base_channels=cfg["disc_base_channels"]).to(device)
    G_ema = copy.deepcopy(G)
    G_ema.eval()

    logger.info(f"Generator params: {sum(p.numel() for p in G.parameters()):,}")
    logger.info(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    # Optimizers
    betas = tuple(cfg["betas"])
    g_opt = torch.optim.Adam(G.parameters(), lr=cfg["learning_rate_g"], betas=betas)
    d_opt = torch.optim.Adam(D.parameters(), lr=cfg["learning_rate_d"], betas=betas)

    # Fixed latent vectors for consistent sample generation
    fixed_z = torch.randn(cfg["sample_grid_size"] ** 2, cfg["latent_dim"], device=device)

    # Resume from checkpoint
    start_step = 0
    best_fid = None
    if resume:
        logger.info(f"Resuming from {resume}")
        start_step, best_fid = load_checkpoint(
            resume, G, D, G_ema, g_opt, d_opt, device
        )
        logger.info(f"Resumed at step {start_step}")

    # Training loop
    grad_accum = cfg["gradient_accumulate_every"]
    aug_ops = cfg.get("aug_ops", ["color", "translation", "cutout"])
    aug_prob = cfg.get("aug_prob", 0.5)
    recon_weight = cfg.get("recon_weight", 0.5)
    ema_decay = cfg.get("ema_decay", 0.999)

    data_iter = iter(dataloader)

    def get_batch():
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        return batch.to(device)

    G.train()
    D.train()

    pbar = tqdm(
        range(start_step, cfg["num_train_steps"]),
        initial=start_step,
        total=cfg["num_train_steps"],
        desc="Training",
    )

    for step in pbar:
        # --- Discriminator update ---
        d_opt.zero_grad()
        d_loss_accum = 0.0

        for _ in range(grad_accum):
            real = get_batch()
            z = torch.randn(real.size(0), cfg["latent_dim"], device=device)
            fake = G(z).detach()

            # Augment
            real_aug = diff_augment(real, aug_ops, aug_prob)
            fake_aug = diff_augment(fake, aug_ops, aug_prob)

            # Discriminator forward
            d_real_logits, d_real_recon = D(real_aug)
            d_fake_logits, _ = D(fake_aug)

            # Loss
            d_loss = hinge_loss_d(d_real_logits, d_fake_logits)
            d_loss = d_loss + recon_weight * reconstruction_loss(d_real_recon, real)
            (d_loss / grad_accum).backward()
            d_loss_accum += d_loss.item() / grad_accum

        d_opt.step()

        # --- Generator update ---
        g_opt.zero_grad()
        g_loss_accum = 0.0

        for _ in range(grad_accum):
            z = torch.randn(cfg["batch_size"], cfg["latent_dim"], device=device)
            fake = G(z)
            fake_aug = diff_augment(fake, aug_ops, aug_prob)
            g_logits, _ = D(fake_aug)

            g_loss = hinge_loss_g(g_logits)
            (g_loss / grad_accum).backward()
            g_loss_accum += g_loss.item() / grad_accum

        g_opt.step()

        # Update EMA
        update_ema(G_ema, G, ema_decay)

        # Logging
        pbar.set_postfix(d_loss=f"{d_loss_accum:.4f}", g_loss=f"{g_loss_accum:.4f}")
        writer.add_scalar("loss/discriminator", d_loss_accum, step)
        writer.add_scalar("loss/generator", g_loss_accum, step)

        # Periodic actions
        if (step + 1) % cfg["sample_every"] == 0:
            generate_samples(G_ema, fixed_z, step + 1, sample_dir, writer)

        if (step + 1) % cfg["save_every"] == 0:
            ckpt_path = checkpoint_dir / f"step_{step + 1:07d}.pt"
            save_checkpoint(ckpt_path, step + 1, G, D, G_ema, g_opt, d_opt, best_fid)
            logger.info(f"Saved checkpoint: {ckpt_path}")

        # Gradient validation (MPS safety check)
        if (step + 1) % 1000 == 0:
            validate_gradients(G, "Generator", step + 1)
            validate_gradients(D, "Discriminator", step + 1)

        # Memory cleanup
        if (step + 1) % 500 == 0:
            mps_empty_cache()

    # Final checkpoint
    save_checkpoint(
        checkpoint_dir / "final.pt",
        cfg["num_train_steps"], G, D, G_ema, g_opt, d_opt, best_fid
    )
    logger.info("Training complete!")
    writer.close()


if __name__ == "__main__":
    main()
