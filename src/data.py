"""
src/data.py — Dataset, transforms, and CV split utilities for Jaguar Re-ID.

All future versions import from here; changes propagate everywhere.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_rgba_as_rgb(path: Path, bg_color=(255, 255, 255)) -> Image.Image:
    """
    Open an image and composite it onto a solid background.

    All jaguar images are RGBA (transparent background from SAM3 segmentation).
    We composite onto white by default so the model sees a clean, consistent
    background rather than undefined transparent pixels.
    """
    img = Image.open(path).convert('RGBA')
    bg  = Image.new('RGB', img.size, bg_color)
    bg.paste(img, mask=img.split()[3])   # use alpha channel as mask
    return bg


def pad_to_square(img: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    """
    Pad image to square with a solid fill color, keeping content centred.

    Aspect ratios in this dataset range from 0.36 to 9.37 — stretching to
    square would heavily distort spot patterns, so we pad instead.
    """
    w, h = img.size
    size = max(w, h)
    new = Image.new('RGB', (size, size), fill)
    new.paste(img, ((size - w) // 2, (size - h) // 2))
    return new


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(mode: str, img_size: int = 224):
    """
    Return transforms for a given mode.

    mode='val'  : resize + centre-crop + normalize (no augmentation)
    mode='train': same as val for v01; augmentation added from v05 onwards
    mode='test' : same as val
    """
    base = [
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(base)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JaguarDataset(Dataset):
    """
    Loads jaguar images from a list of (filename, label) pairs.

    Args:
        filenames: list of image filenames (e.g. 'train_0001.png')
        img_dir:   directory containing the images
        labels:    list of integer class labels (None for test set)
        transform: torchvision transform to apply
        bg_color:  background colour for RGBA compositing (default: white)
    """

    def __init__(self, filenames, img_dir, labels=None, transform=None,
                 bg_color=(255, 255, 255)):
        self.filenames = list(filenames)
        self.img_dir   = Path(img_dir)
        self.labels    = list(labels) if labels is not None else None
        self.transform = transform
        self.bg_color  = bg_color

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.img_dir / self.filenames[idx]
        img  = load_rgba_as_rgb(path, self.bg_color)
        img  = pad_to_square(img, fill=self.bg_color)
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img


# ---------------------------------------------------------------------------
# CV splits
# ---------------------------------------------------------------------------

def get_identity_splits(train_df: pd.DataFrame, n_splits: int = 5):
    """
    Return GroupKFold splits stratified by jaguar identity.

    We group by identity rather than splitting randomly. This prevents
    near-duplicate burst images from the same jaguar leaking across folds,
    which would inflate CV scores and not reflect real generalisation.

    Yields (fold_idx, train_indices, val_indices) tuples.
    """
    gkf    = GroupKFold(n_splits=n_splits)
    groups = train_df['ground_truth'].values
    X      = np.arange(len(train_df))
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
        yield fold, train_idx, val_idx


def encode_labels(series: pd.Series):
    """
    Map string identity names to integer class labels.
    Returns (integer_labels, label_to_idx dict, idx_to_label dict).
    """
    classes     = sorted(series.unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    labels       = series.map(label_to_idx).values
    return labels, label_to_idx, idx_to_label
