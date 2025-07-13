# src/ocr/dataset.py
# Defines the custom Dataset and DataLoader logic for OCR, now with data augmentation.
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
import cv2
import numpy as np
import albumentations as A
# Assuming utils.py is in the same directory or src is in PYTHONPATH
from .utils import load_char_list


class OCRDataset(Dataset):
    """
    Custom PyTorch Dataset for loading OCR data.
    - Reads image paths and labels from an annotation file.
    - Applies specified preprocessing and data augmentation.
    """
    def __init__(self, annotations_file, char_list_path,
                 image_height=32, image_width=256, binarize=False,
                 augmentation_config=None, is_train=True):
        """
        Args:
            annotations_file (str): Path to the annotation file (e.g., CSV with 'filepath', 'text').
            char_list_path (str): Path to the file containing the character set.
            image_height (int): Height to which images will be resized.
            image_width (int): Width to which images will be resized.
            binarize (bool): Whether to binarize images.
            augmentation_config (dict, optional): Configuration for data augmentation.
            is_train (bool): Flag to determine if augmentations should be applied.
        """
        super().__init__()
        print(f"Initializing OCRDataset with annotations from: {annotations_file}")
        self.image_height = image_height
        self.image_width = image_width
        self.binarize = binarize
        self.is_train = is_train

        try:
            # Assuming the CSV has columns 'filepath' and 'text' without a header
            self.annotations = pd.read_csv(annotations_file, header=None, names=['filepath', 'text'], keep_default_na=False)
            print(f"  Loaded {len(self.annotations)} annotations.")
        except Exception as e:
            print(f"Error loading or parsing annotation file {annotations_file}: {e}")
            # If loading fails, create an empty dataframe with the expected columns
            self.annotations = pd.DataFrame(columns=['filepath', 'text'])

        self.char_list = load_char_list(char_list_path)
        self.char_to_int = {char: i + 1 for i, char in enumerate(self.char_list)}
        self.int_to_char = {i + 1: char for i, char in enumerate(self.char_list)}
        self.int_to_char[0] = '<BLANK>'
        print(f"  Character map created. Vocab size: {len(self.char_list)}")

        # --- Setup Augmentation Pipeline ---
        self.augmenter = None
        if self.is_train and augmentation_config and augmentation_config.get('enable', False):
            print("  Data augmentation is ENABLED for this dataset.")
            self.augmenter = self._create_augmentation_pipeline(augmentation_config)
        else:
            print("  Data augmentation is DISABLED for this dataset.")

    def _create_augmentation_pipeline(self, config):
        """Creates the Albumentations augmentation pipeline from config."""
        transforms = []
        if config.get('rotate_limit'):
            transforms.append(A.SafeRotate(limit=config['rotate_limit'], p=0.5, border_mode=cv2.BORDER_CONSTANT))
        if config.get('scale_limit'):
            transforms.append(A.LongestMaxSize(max_size=self.image_width, interpolation=cv2.INTER_LINEAR, p=0.5))
            # transforms.append(A.RandomScale(scale_limit=config['scale_limit'], p=0.5))
        if config.get('shear_limit'):
             transforms.append(A.Affine(shear=config['shear_limit'], p=0.3, mode=cv2.BORDER_CONSTANT))
        if config.get('grid_distortion', {}).get('enable'):
            params = config['grid_distortion']
            transforms.append(A.GridDistortion(
                num_steps=params.get('num_steps', 5),
                distort_limit=params.get('distort_limit', 0.3),
                p=0.5, border_mode=cv2.BORDER_CONSTANT
            ))
        if config.get('brightness_contrast', {}).get('enable'):
            params = config['brightness_contrast']
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=params.get('brightness_limit', 0.2),
                contrast_limit=params.get('contrast_limit', 0.2),
                p=0.5
            ))
        if config.get('blur', {}).get('enable'):
            transforms.append(A.Blur(blur_limit=config['blur'].get('blur_limit', 3), p=0.3))
        if config.get('gaussian_noise', {}).get('enable'):
            transforms.append(A.GaussNoise(var_limit=config['gaussian_noise'].get('var_limit', [10.0, 50.0]), p=0.3))

        return A.Compose(transforms)


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError("Index out of range")

        record = self.annotations.iloc[idx]
        image_path = record['filepath']
        text_label = record['text']

        try:
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None: raise ValueError(f"Image not found or unreadable at {image_path}")

            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply augmentations if enabled
            if self.augmenter:
                augmented = self.augmenter(image=image)
                image = augmented['image']

            # Binarize after augmentation if enabled
            if self.binarize:
                _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Resize to fixed size (H, W)
            # Note: Albumentations can handle resizing, but doing it here ensures consistency
            # especially after geometric transforms that might change image size.
            image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)

            # Normalize and add channel dimension
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            image_tensor = torch.from_numpy(image)

        except Exception as e:
            print(f"Warning: Error processing image {image_path}: {e}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self)) if len(self) > 1 else (None, None)

        # Encode text label
        encoded_text = [self.char_to_int.get(char, 0) for char in text_label] # Use 0 for unknown chars
        label_tensor = torch.LongTensor(encoded_text)

        return image_tensor, label_tensor

def ocr_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if not batch: return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    images, labels = zip(*batch)
    images_stacked = torch.stack(images, 0)
    label_lengths = torch.IntTensor([len(label) for label in labels])
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    return images_stacked, labels_padded, None, label_lengths # Return None for image_widths for now

if __name__ == '__main__':
    print("\n--- Testing OCR Dataset with Augmentations ---")
    # This test requires a dummy setup similar to the previous version
    # and also a dummy augmentation config.
    test_dir = "temp_ocr_aug_test"
    img_dir = os.path.join(test_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    annotations = {'filepath': [os.path.abspath(os.path.join(img_dir, 'img1.png'))], 'text': ['augmented']}
    annotations_path = os.path.join(test_dir, "annotations.csv")
    pd.DataFrame(annotations).to_csv(annotations_path, index=False)

    cv2.imwrite(os.path.join(img_dir, "img1.png"), np.zeros((40, 150, 3), dtype=np.uint8))

    char_list_path = os.path.join(test_dir, "chars.txt")
    with open(char_list_path, "w") as f: f.write("a\nu\ng\nm\ne\nn\nt\nd\n")

    aug_config = {
        'enable': True,
        'rotate_limit': 10,
        'scale_limit': 0.15,
        'grid_distortion': {'enable': True},
        'brightness_contrast': {'enable': True},
        'blur': {'enable': True},
        'gaussian_noise': {'enable': True}
    }

    print("\n--- Initializing Dataset with Augmentations ENABLED ---")
    try:
        aug_dataset = OCRDataset(
            annotations_file=annotations_path,
            char_list_path=char_list_path,
            image_height=32, image_width=200,
            augmentation_config=aug_config,
            is_train=True
        )

        # Fetch one sample to see if it works
        img_tensor, _ = aug_dataset[0]
        print(f"Sample 0 from augmented dataset shape: {img_tensor.shape}")
        # We can't easily check if augmentation was applied without visualizing,
        # but if it runs without error, the pipeline is working.
        if img_tensor.shape == (1, 32, 200):
            print("Augmented dataset test PASSED.")
        else:
            print("Augmented dataset test FAILED. Shape mismatch.")

    except Exception as e:
        print(f"An error occurred during augmented dataset test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import shutil
        if os.path.exists(test_dir): shutil.rmtree(test_dir)

    print("\n--- Dataset Augmentation Script Finished ---")
