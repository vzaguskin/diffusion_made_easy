"""MNIST data via torchvision, wrapped as a LightningDataModule.

Key choice — *normalization* (theory.md §3, variance-preserving):
    The forward process assumes ``Var(x_0) ≈ 1``. MNIST pixel values in [0, 1]
    have variance ≈ 0.095 (most pixels are black). We normalize with MNIST's
    empirical mean/std (0.1307, 0.3081) so the data has zero mean and unit
    variance. This keeps the variance-preserving math of the schedule correct.

Split: standard torchvision MNIST gives 60k train + 10k test. We hold out the
last 5k of train as validation → 55k train / 5k val / 10k test.
"""

from __future__ import annotations

from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# MNIST empirical mean/std — the canonical normalization constants.
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


class MNISTDataModule(LightningDataModule):
    """Downloads MNIST, normalizes it, and serves train/val/test loaders."""

    def __init__(
        self,
        data_dir: str = "data",
        train_batch_size: int = 128,
        eval_batch_size: int = 256,
        num_workers: int = 4,
        val_size: int = 5000,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.val_size = val_size

        # ToTensor scales uint8 [0,255] -> float [0,1]; Normalize gives zero-mean/unit-var.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])

        # These are populated in setup().
        self.mnist_train: datasets.MNIST | None = None
        self.mnist_val: datasets.MNIST | None = None
        self.mnist_test: datasets.MNIST | None = None

    @classmethod
    def from_config(cls, cfg: Any) -> "MNISTDataModule":
        """Build from the ``data`` section of the config."""
        d = cfg.data
        return cls(
            data_dir=getattr(d, "data_dir", "data"),
            train_batch_size=getattr(d, "train_batch_size", 128),
            eval_batch_size=getattr(d, "eval_batch_size", 256),
            num_workers=getattr(d, "num_workers", 4),
            val_size=getattr(d, "val_size", 5000),
        )

    def setup(self, stage: str | None = None) -> None:
        """Download (if needed) and split MNIST. Cached after the first call."""
        if self.mnist_train is not None:
            return  # idempotent
        full_train = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=self.transform
        )
        # Split off val_size from the end of the 60k train set; remaining is train.
        train_size = len(full_train) - self.val_size
        self.mnist_train, self.mnist_val = random_split(
            full_train,
            [train_size, self.val_size],
            generator=torch.Generator().manual_seed(42),  # deterministic split
        )
        self.mnist_test = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=self.transform
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def shape(self) -> tuple[int, int, int]:
        """Image shape as (C, H, W)."""
        return (1, 28, 28)
