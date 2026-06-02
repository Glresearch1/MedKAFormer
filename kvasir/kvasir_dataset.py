from pathlib import Path
from typing import Optional, Tuple

import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMG_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class KvasirDataset(Dataset):
    """Kvasir-style image folder dataset.

    Expected layout:

        root_dir/
            class_0/
                image_001.jpg
            class_1/
                image_002.jpg
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_dir}")

        self.classes = sorted(
            path.name for path in self.root_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
        if not self.classes:
            raise ValueError(f"No class folders found in: {self.root_dir}")

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMG_EXTENSIONS:
                    self.image_paths.append(image_path)
                    self.labels.append(self.class_to_idx[class_name])

        if not self.image_paths:
            raise ValueError(f"No images found in class folders under: {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def build_kvasir_transforms(
    resize_size: int = 256,
    image_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (resize_size, resize_size),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (resize_size, resize_size),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform


def create_kvasir_dataloaders(
    train_dir,
    test_dir,
    val_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    resize_size: int = 256,
    image_size: int = 224,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    train_transform, eval_transform = build_kvasir_transforms(resize_size=resize_size, image_size=image_size)

    train_dataset = KvasirDataset(root_dir=train_dir, transform=train_transform)
    test_dataset = KvasirDataset(root_dir=test_dir, transform=eval_transform)
    val_dataset = KvasirDataset(root_dir=val_dir, transform=eval_transform) if val_dir else test_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
