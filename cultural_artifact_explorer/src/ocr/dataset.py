# src/ocr/dataset.py
# Dataset and collate function for CRNN/CTC OCR training.

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from src.ocr.utils import load_char_list, preprocess_image_for_ocr

class OCRDataset(Dataset):
    def __init__(self, annotations_file, img_dir, char_list_path,
                 image_height=32, image_width=256, binarize=False):
        """
        Args:
            annotations_file (str): Path to CSV with image filename/text (with header: filename or filepath, text)
            img_dir (str): Path to images root dir
            char_list_path (str): Path to char list (should have <BLANK> at index 0)
            image_height (int): Height to resize images to
            image_width (int): Width to resize images to
            binarize (bool): Whether to binarize images (Otsu)
        """
        self.img_dir = img_dir
        self.char_list = load_char_list(char_list_path)
        self.char_to_ix = {c: i for i, c in enumerate(self.char_list)}
        self.ix_to_char = {i: c for i, c in enumerate(self.char_list)}
        self.image_height = image_height
        self.image_width = image_width
        self.binarize = binarize

        df = pd.read_csv(annotations_file)
        # Accept both possible column names for filename
        filename_col = None
        for col in ['filename', 'filepath']:
            if col in df.columns:
                filename_col = col
                break
        if filename_col is None:
            raise ValueError(
                f"Neither 'filename' nor 'filepath' found in {annotations_file}. Found columns: {list(df.columns)}"
            )
        df = df.dropna(subset=[filename_col, "text"])
        self.samples = []
        for _, row in df.iterrows():
            fname = str(row[filename_col])
            label = str(row["text"])
            self.samples.append((fname, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        import os
        if os.path.isabs(fname):
            img_path = fname
        else:
            img_path = os.path.join(self.img_dir, fname)
        try:
            img = preprocess_image_for_ocr(
                img_path,
                target_size=(self.image_width, self.image_height),
                binarize=self.binarize,
            )
        except Exception as e:
            print(f"Warning: Failed to process {img_path}, skipping. Error: {e}")
            blank_img = np.zeros((1, self.image_height, self.image_width), dtype=np.float32)
            return torch.from_numpy(blank_img), torch.tensor([0], dtype=torch.long), "<BLANK>", 1
        # Encode label as indices, using <BLANK> (0) for OOV chars
        label_idx = [self.char_to_ix.get(char, 0) for char in label]
        label_len = len(label_idx)
        return torch.from_numpy(img), torch.tensor(label_idx, dtype=torch.long), label, label_len


def ocr_collate_fn(batch):
    """
    Collate function for variable-length OCR targets (for CTC).
    batch: list of (image, label_idx, label_str, label_len)
    Returns:
        images: (B, 1, H, W)
        labels: (sum_label_lens,)
        label_strs: list[str]
        label_lens: (B,)
    """
    images, labels, label_strs, label_lens = zip(*batch)
    images = torch.stack(images, 0)
    labels_concat = torch.cat([l for l in labels], 0)
    label_lens = torch.tensor(label_lens, dtype=torch.long)
    return images, labels_concat, label_strs, label_lens

if __name__ == "__main__":
    ann = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/annotations.csv"
    imgs = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/images"
    clist = "/Users/paarthmiglani/PycharmProjects/manualagent/cultural_artifact_explorer/scripts/data/ocr/char_list.txt"
    dset = OCRDataset(ann, imgs, clist)
    print("Sample:", dset[0])
