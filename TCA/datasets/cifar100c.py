"""CIFAR-100-C loader for the Table-2 reproduction (TCA paper).

CIFAR-100-C is NOT a file-on-disk + JSON-split dataset like the cross-dataset (CD)
benchmark. Each corruption is a single NumPy array `<corruption>.npy` of shape
(50000, 32, 32, 3) — the 10 000-image CIFAR-100 *test* set re-rendered at 5 severities
and stacked: rows [(s-1)*10000 : s*10000] are severity `s` (s = 1..5). `labels.npy`
(shape (50000,)) holds the matching labels (the same 10 000 test labels tiled 5x).

Because there are no per-image paths, this can't go through `DatasetWrapper`
(which does `read_image(item.impath)`). We expose a thin `torch.utils.data.Dataset`
that turns each uint8 row into a PIL image and applies the CLIP `preprocess`, exactly
as the CD pipeline applies `preprocess` to its PIL images.

Table 2 only reports Contrast / Snow / Brightness, but any corruption file present on
disk works (see CORRUPTIONS for the full CIFAR-100-C set).

Prompts: the official OpenAI CLIP CIFAR-100 18-template ensemble (see prompts docs).
All three Table-2 methods (CLIP, EViT, TCA) read `template` from here, so they share
this ensemble automatically. (The CLIP 80-prompt variant is selected at run time via
`clip_zeroshot.py --imagenet-ensemble`, which overrides this template.)
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# CIFAR-100 fine-label names in label-index order (0..99) — matches CIFAR-100-C labels.npy.
CIFAR100_CLASSNAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm',
]

# Official OpenAI CLIP CIFAR-100 prompt templates (18-template ensemble).
templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

# Full CIFAR-100-C corruption set (Table 2 uses only the first three).
CORRUPTIONS = [
    'contrast', 'snow', 'brightness',
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'frost', 'fog',
    'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise',
    'gaussian_blur', 'spatter', 'saturate',
]

IMAGES_PER_SEVERITY = 10000


class CIFAR100C(Dataset):
    """One (corruption, severity) slice of CIFAR-100-C, preprocessed for CLIP."""

    def __init__(self, root, corruption, severity, transform):
        self.transform = transform
        data_dir = os.path.join(root, 'CIFAR-100-C')
        npy = os.path.join(data_dir, f'{corruption}.npy')
        if not os.path.exists(npy):
            raise FileNotFoundError(
                f"CIFAR-100-C corruption file not found: {npy}\n"
                f"Run scripts-t2/download_cifar100c.sh first."
            )

        images = np.load(npy)                                   # (50000, 32, 32, 3) uint8
        labels = np.load(os.path.join(data_dir, 'labels.npy'))  # (50000,)

        s = int(severity)
        if not 1 <= s <= 5:
            raise ValueError(f"severity must be 1..5, got {severity}")
        lo, hi = (s - 1) * IMAGES_PER_SEVERITY, s * IMAGES_PER_SEVERITY
        self.images = images[lo:hi]
        self.labels = labels[lo:hi]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])  # uint8 HxWxC -> PIL RGB
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])
