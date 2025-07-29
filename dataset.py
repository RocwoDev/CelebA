from pathlib import Path
from torchvision.io import decode_image
from torch.utils.data import Dataset
import zipfile
import torch


class CelebA(Dataset):
    eval_partition_path = Path(__file__).parent / "Eval" / "list_eval_partition.txt"
    identity_anno_path = Path(__file__).parent / "Anno" / "identity_CelebA.txt"
    images_repo_path = Path(__file__).parent / "Images.zip"

    def __init__(self, eval_str: str, transform=None, target_transform=None):
        """
        Generate a dataset for the CelebA dataset.
        Args:
            eval_str (str): "train" for training, "val" for validation, "test" for testing
            size_str (str): "original" for original images, "cropped" for cropped images
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the label.
        """
        eval_number = 0 if eval_str == "train" else 1 if eval_str == "val" else 2
        self.images_repo = CelebA.images_repo_path

        self.images_names = []
        with open(CelebA.eval_partition_path, "r") as f:
            for line in f.readlines():
                image_name, image_eval = line.strip().split(" ")
                if image_eval == str(eval_number):
                    self.images_names.append(image_name)

        self.images_identities = dict()
        with open(CelebA.identity_anno_path, "r") as f:
            for line in f.readlines():
                image_name, identity = line.strip().split(" ")
                self.images_identities[image_name] = int(identity)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        if idx >= len(self.images_names) or idx < 0:
            raise IndexError(f"Index {idx} is out of bounds for dataset size ({len(self.images_names)})")
        image_name = self.images_names[idx]

        with zipfile.ZipFile(self.images_repo, 'r') as z:
            with z.open(image_name) as f:
                image_data = f.read()

        if not image_data:
            raise ValueError(f"Empty data for image: {image_name}")

        # Convert bytes to writable buffer and then to 1D tensor
        image_tensor = torch.tensor(bytearray(image_data), dtype=torch.uint8)
        image = decode_image(image_tensor)

        identity = self.images_identities[image_name]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            identity = self.target_transform(identity)
        return image, identity
