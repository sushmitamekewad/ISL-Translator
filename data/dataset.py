import io
import os
import zipfile
import random
from typing import Tuple, List, Dict, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# Supported image extensions
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

class ISLZipDataset(Dataset):
    """
    PyTorch Dataset that reads ISL (Indian Sign Language) images directly from a ZIP file
    WITHOUT extracting the archive to disk.
    
    Structure expected inside ZIP:
        <class_name>/<image_name>.jpg (e.g. 'a/A1.jpg', 'b/B12.jpg', ...)
    """
    def __init__(
        self,
        zip_path: str,
        transform: Optional[transforms.Compose] = None,
        samples: Optional[List[Tuple[str, int]]] = None,
        classes: Optional[List[str]] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.zip_path = os.path.abspath(zip_path)
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"ZIP file not found at: {self.zip_path}")
            
        self.transform = transform
        self._zip_file: Optional[zipfile.ZipFile] = None  # Lazily opened per worker

        if samples is not None and classes is not None and class_to_idx is not None:
            self.samples = samples
            self.classes = classes
            self.class_to_idx = class_to_idx
        else:
            self._scan_zip()

    def _get_zip(self) -> zipfile.ZipFile:
        """Lazily initialize the zipfile handle for DataLoader multiprocessing safety."""
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.zip_path, mode='r')
        return self._zip_file

    def _scan_zip(self):
        """Scans the zip archive to index image paths and extract class labels."""
        with zipfile.ZipFile(self.zip_path, mode='r') as z:
            namelist = z.namelist()
            
            # Filter valid image paths, ignore MacOS artifacts and top-level dirs
            raw_entries = [
                name for name in namelist
                if not name.startswith('__MACOSX')
                and not name.startswith('.')
                and name.lower().endswith(VALID_EXTENSIONS)
            ]

            # Determine classes from directory names
            classes_set = set()
            for path in raw_entries:
                parts = path.strip('/').split('/')
                if len(parts) >= 2:
                    classes_set.add(parts[-2])

            self.classes = sorted(list(classes_set))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            # Build list of (internal_zip_path, label_idx)
            self.samples = []
            for path in raw_entries:
                parts = path.strip('/').split('/')
                if len(parts) >= 2:
                    cls_name = parts[-2]
                    if cls_name in self.class_to_idx:
                        self.samples.append((path, self.class_to_idx[cls_name]))

        print(f"[ISLZipDataset] Indexed {len(self.samples)} images across {len(self.classes)} classes from {os.path.basename(self.zip_path)}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_internal_path, label = self.samples[idx]
        zf = self._get_zip()

        # Read binary data straight from archive into memory
        raw_bytes = zf.read(img_internal_path)
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_sample_image(self, idx: int) -> Tuple[Image.Image, str]:
        """Utility to get raw PIL image and class label without tensor transforms."""
        img_internal_path, label = self.samples[idx]
        zf = self._get_zip()
        raw_bytes = zf.read(img_internal_path)
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return image, self.classes[label]

    def __del__(self):
        if self._zip_file is not None:
            try:
                self._zip_file.close()
            except Exception:
                pass


def get_isl_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Standard ImageNet normalization and sign-safe data augmentations.
    Note: RandomHorizontalFlip is deliberately omitted because ISL hand gestures
    have semantic meaning tied to hand orientation/handedness.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def create_stratified_split(
    base_dataset: ISLZipDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Splits samples stratified by class without writing anything to disk."""
    random.seed(seed)
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for sample in base_dataset.samples:
        by_class.setdefault(sample[1], []).append(sample)

    train_samples, val_samples, test_samples = [], [], []

    for cls_idx, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_samples.extend(items[:n_train])
        val_samples.extend(items[n_train:n_train + n_val])
        test_samples.extend(items[n_train + n_val:])

    return train_samples, val_samples, test_samples


def build_dataloaders(
    zip_path: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Convenience factory to get Train, Val, and Test DataLoaders directly from the ZIP."""
    train_tf, eval_tf = get_isl_transforms(img_size=img_size)
    
    # Base dataset for scanning metadata
    base = ISLZipDataset(zip_path=zip_path)
    train_s, val_s, test_s = create_stratified_split(base, seed=seed)

    train_ds = ISLZipDataset(zip_path, transform=train_tf, samples=train_s, classes=base.classes, class_to_idx=base.class_to_idx)
    val_ds = ISLZipDataset(zip_path, transform=eval_tf, samples=val_s, classes=base.classes, class_to_idx=base.class_to_idx)
    test_ds = ISLZipDataset(zip_path, transform=eval_tf, samples=test_s, classes=base.classes, class_to_idx=base.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, base.classes
