"""Export trained generator to ONNX format for deployment."""

from pathlib import Path

import click
import torch

from model.generator import Generator


@click.command()
@click.option("--checkpoint", "-c", required=True, help="Path to training checkpoint.")
@click.option("--output", "-o", default="generator.onnx", help="Output ONNX path.")
@click.option("--latent-dim", default=256, help="Latent vector dimension.")
@click.option("--base-channels", default=256, help="Generator base channels.")
@click.option("--also-trace", is_flag=True, help="Also export torch.jit.trace version.")
def main(checkpoint: str, output: str, latent_dim: int, base_channels: int, also_trace: bool):
    """Export the EMA generator from a checkpoint to ONNX."""
    # Load model
    G = Generator(latent_dim=latent_dim, base_channels=base_channels)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    G.load_state_dict(ckpt["ema_generator"])
    G.eval()

    # Export ONNX
    dummy_z = torch.randn(1, latent_dim)
    torch.onnx.export(
        G,
        dummy_z,
        output,
        opset_version=17,
        input_names=["latent_vector"],
        output_names=["image"],
        dynamic_axes={
            "latent_vector": {0: "batch"},
            "image": {0: "batch"},
        },
    )
    click.echo(f"Exported ONNX model to {output}")

    # Optional: torch.jit.trace
    if also_trace:
        traced = torch.jit.trace(G, dummy_z)
        trace_path = Path(output).with_suffix(".pt")
        traced.save(str(trace_path))
        click.echo(f"Exported traced model to {trace_path}")


if __name__ == "__main__":
    main()
