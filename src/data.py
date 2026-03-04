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


def get_transforms(mode: str, img_size: int = 224, augment_level: str = 'light',
                   cached: bool = False):
    """
    Return transforms for a given mode.

    mode='val' / 'test': resize + centre-crop + normalize (no augmentation)
    mode='train':        adds augmentation controlled by augment_level:
        'none'  - identical to val (used in v01)
        'light' - horizontal flip + mild colour jitter (v02 default)
        'heavy' - adds random erasing, rotation, aggressive jitter (v05+)

    cached=True: omit the Resize step. Use this when images come from
        build_image_cache(), which pre-applies Resize(img_size+32).

    Note on horizontal flips for jaguar re-ID:
        Left and right flanks have DIFFERENT spot patterns, so a horizontal
        flip creates a plausible-looking but WRONG identity. We include it
        as a regulariser with low probability (0.3) since it forces the
        model to use texture geometry rather than global layout, but be
        aware it can slightly hurt performance on viewpoint-constrained data.
    """
    resize_step = [] if cached else [transforms.Resize(img_size + 32)]

    val_transforms = resize_step + [
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    if mode in ('val', 'test') or augment_level == 'none':
        return transforms.Compose(val_transforms)

    if augment_level == 'light':
        train_transforms = resize_step + [
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    elif augment_level == 'heavy':
        heavy_resize = [] if cached else [transforms.Resize(img_size + 64)]
        train_transforms = heavy_resize + [
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    else:
        raise ValueError(f'Unknown augment_level: {augment_level}')

    return transforms.Compose(train_transforms)


# ---------------------------------------------------------------------------
# Image cache (optional, for eliminating disk I/O bottleneck)
# ---------------------------------------------------------------------------

def build_image_cache(filenames, img_dir, img_size=224,
                      bg_color=(255, 255, 255), verbose=True):
    """
    Pre-load and pre-process all images into a dict stored in RAM.

    Applies load_rgba_as_rgb + pad_to_square + Resize(img_size + 32) once.
    Subsequent dataset __getitem__ calls skip disk I/O entirely — only the
    cheap random transforms (crop, jitter, ToTensor, Normalize) fire per call.

    RAM cost: len(filenames) * (img_size+32)^2 * 3 bytes
              e.g. 1895 * 256^2 * 3 ≈ 375 MB for the full jaguar train set.

    Args:
        filenames: list of image filenames to cache
        img_dir:   directory containing the images
        img_size:  target (img_size+32) for pre-resize before torchvision crops
        bg_color:  background for RGBA compositing
        verbose:   print progress

    Returns:
        dict mapping filename -> RGB PIL.Image at size (img_size+32)
    """
    cache     = {}
    pre_size  = img_size + 32   # matches the Resize() step in get_transforms
    img_dir   = Path(img_dir)
    unique    = list(dict.fromkeys(filenames))   # preserve order, deduplicate

    if verbose:
        print(f'  Building image cache ({len(unique)} images -> RAM)...')

    for i, fname in enumerate(unique):
        img = load_rgba_as_rgb(img_dir / fname, bg_color)
        img = pad_to_square(img, fill=bg_color)
        img = img.resize((pre_size, pre_size), Image.BILINEAR)
        cache[fname] = img
        if verbose and (i + 1) % 200 == 0:
            print(f'    {i+1}/{len(unique)} cached')

    if verbose:
        import sys
        mb = sum(c.size[0] * c.size[1] * 3 for c in cache.values()) / 1e6
        print(f'  Cache complete: {len(cache)} images, ~{mb:.0f} MB')
    return cache


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JaguarDataset(Dataset):
    """
    Loads jaguar images from a list of (filename, label) pairs.

    Args:
        filenames:   list of image filenames (e.g. 'train_0001.png')
        img_dir:     directory containing the images
        labels:      list of integer class labels (None for test set)
        transform:   torchvision transform to apply
        bg_color:    background colour for RGBA compositing (default: white)
        image_cache: optional dict {filename -> PIL Image} from build_image_cache.
                     When provided, disk I/O is skipped entirely — the cached
                     PIL image is used directly as the transform input.
                     Note: cached images are already Resize(img_size+32)-ed, so
                     get_transforms() must NOT include a Resize step when using
                     the cache (use get_transforms with skip_resize=True, or pass
                     transforms that start from CenterCrop/RandomCrop).
    """

    def __init__(self, filenames, img_dir, labels=None, transform=None,
                 bg_color=(255, 255, 255), image_cache=None):
        self.filenames   = list(filenames)
        self.img_dir     = Path(img_dir)
        self.labels      = list(labels) if labels is not None else None
        self.transform   = transform
        self.bg_color    = bg_color
        self.image_cache = image_cache   # {filename: PIL Image} or None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        if self.image_cache is not None and fname in self.image_cache:
            img = self.image_cache[fname]   # pre-composited, padded, resized PIL
        else:
            path = self.img_dir / fname
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
