"""Data pipeline for traffic-sign classification with the ViT.

Author: Krunal Vaghela

Uses the German Traffic Sign Recognition Benchmark (GTSRB, 43 classes) through torchvision, which downloads it
on first use. Any folder of images arranged as <root>/<class_name>/*.png works too (--data-dir).
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

GTSRB_MEAN, GTSRB_STD = (0.34, 0.31, 0.32), (0.27, 0.26, 0.27)


def build_transforms(image_size, train):
    t = [transforms.Resize((image_size, image_size))]
    if train:
        # no horizontal flips: a mirrored "keep left" sign would become "keep right"
        t += [transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)]
    t += [transforms.ToTensor(), transforms.Normalize(GTSRB_MEAN, GTSRB_STD)]
    return transforms.Compose(t)


def build_loaders(image_size=64, batch_size=128, data_root="data", data_dir=None, workers=4):
    if data_dir:                                            # your own <root>/<class>/*.png folders
        root = Path(data_dir)
        train = datasets.ImageFolder(root / "train", build_transforms(image_size, True))
        test = datasets.ImageFolder(root / "test", build_transforms(image_size, False))
        classes = train.classes
    else:
        train = datasets.GTSRB(data_root, split="train", transform=build_transforms(image_size, True), download=True)
        test = datasets.GTSRB(data_root, split="test", transform=build_transforms(image_size, False), download=True)
        classes = [str(i) for i in range(43)]
    pin = torch.cuda.is_available()
    return (DataLoader(train, batch_size, shuffle=True, num_workers=workers, pin_memory=pin, drop_last=True),
            DataLoader(test, batch_size, shuffle=False, num_workers=workers, pin_memory=pin),
            classes)
