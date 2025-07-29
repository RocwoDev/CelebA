# CelebA Dataset Implementation

This project provides a PyTorch-compatible dataset class for the CelebA dataset, enabling access to images and their associated identities.

## Overview

The `CelebA` class in `dataset.py` is a custom PyTorch `Dataset` implementation that loads images from a zip archive (`Images.zip`) and retrieves image identities from annotation files. It supports training, validation, and testing splits, with optional transforms for images and labels. The `display_sample.py` script visualizes sample images with their identities.

## Files

- **dataset.py**: Defines the `CelebA` class for loading and processing the CelebA dataset.
- **display_sample.py**: Displays a 3x3 grid of sample images with their identities using Matplotlib.
- **requirements.txt**: Lists required dependencies (`torch`, `torchvision`).

## Clone Repository

To clone the repository, use:
```bash
git lfs clone https://github.com/RocwoDev/CelebA.git
```

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

Required files (not included):
- `Eval/list_eval_partition.txt`: Specifies train/val/test splits.
- `Anno/identity_CelebA.txt`: Maps image names to identities.
- `Images.zip`: Zip archive containing CelebA images.

## Usage

1. **Setup Dataset**:
   Ensure the required files are in the correct directory structure:
   ```
   .
   ├── Anno
   │   └── identity_CelebA.txt
   ├── Eval
   │   └── list_eval_partition.txt
   ├── Images.zip
   ├── dataset.py
   ├── display_sample.py
   └── requirements.txt
   ```

2. **Run Sample Visualization**:
   ```bash
   python display_sample.py
   ```
   This displays 9 sample images from the training set with their identities.

3. **Use in PyTorch**:
   ```python
   from CelebA import CelebA
   train_dataset = CelebA("train", transform=None, target_transform=None)
   img, identity = train_dataset[0]  # Access image and identity
   ```

## Notes

- Ensure `Images.zip` contains images with names matching those in `list_eval_partition.txt` and `identity_CelebA.txt`.
- Error handling in `__getitem__` raises a `ValueError` for empty image data but does not handle missing or corrupted images.
- Images are loaded as tensors using `torchvision.io.decode_image` and may require transforms for model compatibility.

## Dependencies

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib (for `display_sample.py`, it's not required for the dataset class itself)
