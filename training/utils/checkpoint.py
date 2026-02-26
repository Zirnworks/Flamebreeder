"""Checkpoint save/load utilities."""

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    step: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    ema_generator: torch.nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    best_fid: float | None = None,
):
    """Save a training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "step": step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "ema_generator": ema_generator.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "best_fid": best_fid,
    }, path)


def load_checkpoint(
    path: str | Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    ema_generator: torch.nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float | None]:
    """Load a training checkpoint.

    Returns (step, best_fid).
    """
    data = torch.load(path, map_location=device, weights_only=False)

    generator.load_state_dict(data["generator"])
    discriminator.load_state_dict(data["discriminator"])
    ema_generator.load_state_dict(data["ema_generator"])
    g_optimizer.load_state_dict(data["g_optimizer"])
    d_optimizer.load_state_dict(data["d_optimizer"])

    return data["step"], data.get("best_fid")
