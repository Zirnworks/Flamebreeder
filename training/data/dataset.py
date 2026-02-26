"""PyTorch Dataset for fractal flame images."""

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FlameDataset(Dataset):
    """Dataset of fractal flame images stored as PNG files."""

    def __init__(self, root: str | Path, image_size: int = 512):
        self.root = Path(root)
        self.image_size = image_size

        # Collect all PNG images
        self.paths = sorted(
            list(self.root.glob("*.png")) + list(self.root.glob("*.jpg"))
        )
        if not self.paths:
            raise ValueError(f"No images found in {self.root}")

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)
