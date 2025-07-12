# src/ocr/dataset.py
# Defines the custom Dataset and DataLoader logic for OCR.

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd # Using pandas to easily read annotation files (e.g., csv, tsv)
# from PIL import Image
import numpy as np

# Assuming utils.py is in the same directory or src is in PYTHONPATH
from .utils import preprocess_image_for_ocr, load_char_list

class OCRDataset(Dataset):
    """
    Custom PyTorch Dataset for loading OCR data.
    Assumes an annotation file that maps image filenames to their text labels.
    """
    def __init__(self, annotations_file, img_dir, char_list_path,
                 image_height=32, image_width=128, binarize=False):
        """
        Args:
            annotations_file (str): Path to the annotation file (e.g., a CSV with 'filename' and 'text' columns).
            img_dir (str): Directory where images are stored.
            char_list_path (str): Path to the file containing the character set.
            image_height (int): Height to which images will be resized.
            image_width (int): Width to which images will be resized.
            binarize (bool): Whether to binarize images during preprocessing.
        """
        super().__init__()
        print(f"Initializing OCRDataset with annotations from: {annotations_file}")
        self.img_dir = img_dir
        self.image_height = image_height
        self.image_width = image_width
        self.binarize = binarize

        # Load annotations
        # Assuming a simple CSV/TSV format. Adjust separator if needed.
        try:
            # self.annotations = pd.read_csv(annotations_file, sep='\t', header=None, names=['filename', 'text'], keep_default_na=False)
            self.annotations = pd.read_csv(annotations_file, keep_default_na=False) # Assumes header 'filename', 'text'
            print(f"  Loaded {len(self.annotations)} annotations.")
        except Exception as e:
            print(f"Error loading or parsing annotation file {annotations_file}: {e}")
            self.annotations = pd.DataFrame(columns=['filename', 'text']) # Empty dataframe

        # Load character list and create mapping
        self.char_list = load_char_list(char_list_path)
        # Create char -> int and int -> char mappings
        # The blank token is implicitly handled by CTC loss (usually at index 0),
        # so our mapping should not include a blank token if the char_list file doesn't.
        # Let's shift indices by 1 to reserve 0 for blank.
        self.char_to_int = {char: i + 1 for i, char in enumerate(self.char_list)}
        self.int_to_char = {i + 1: char for i, char in enumerate(self.char_list)}
        # Add entry for the CTC blank token
        self.int_to_char[0] = '<BLANK>'
        print(f"  Character map created. Vocab size: {len(self.char_list)}, Num classes for CTC: {len(self.char_list) + 1}")


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Returns one sample of data: an image and its corresponding encoded label.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        record = self.annotations.iloc[idx]
        image_filename = record['filename']
        text_label = record['text']

        image_path = os.path.join(self.img_dir, image_filename)

        # Preprocess the image
        try:
            # Note: preprocess_image_for_ocr from utils returns (C, H, W)
            # We resize to a fixed W here. A more advanced version might keep aspect ratio and pad.
            image = preprocess_image_for_ocr(
                image_path,
                target_size=(self.image_width, self.image_height), # (W, H) for this function
                binarize=self.binarize
            )
            # Convert to torch tensor
            image_tensor = torch.from_numpy(image)
        except Exception as e:
            print(f"Warning: Error processing image {image_path}: {e}. Skipping sample.")
            # Return a dummy sample or the next valid one
            # For simplicity, we'll just return None and handle in collate_fn or re-raise
            # return None, None
            # Let's try to return the next item
            return self.__getitem__((idx + 1) % len(self))


        # Encode the text label into integers
        encoded_text = []
        for char in text_label:
            # If char is not in our vocab, we can ignore it or map to an <UNK> token
            if char in self.char_to_int:
                encoded_text.append(self.char_to_int[char])
            # else: print(f"Warning: Character '{char}' in '{text_label}' not in char_list. Ignoring.")

        label_tensor = torch.LongTensor(encoded_text)

        return image_tensor, label_tensor


def ocr_collate_fn(batch):
    """
    Custom collate function to handle batching of OCR data.
    It pads images (if they have variable width, though not in this version)
    and pads the encoded text labels.
    Args:
        batch (list): A list of tuples, where each tuple is (image_tensor, label_tensor).
    Returns:
        tuple: A tuple containing:
            - images_padded (torch.Tensor): Padded batch of images.
            - labels_padded (torch.Tensor): Padded batch of labels.
            - image_widths (torch.IntTensor): Widths of each image (as sequence length for RNN).
            - label_lengths (torch.IntTensor): Lengths of each text label.
    """
    # Filter out None samples that might have resulted from loading errors
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if not batch:
        # Return empty tensors if the whole batch failed
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

    images, labels = zip(*batch)

    # --- Handle Images ---
    # In this implementation, all images are resized to the same width, so no padding is needed.
    # If using variable width, you would pad them here.
    # For now, we just stack them.
    images_stacked = torch.stack(images, 0)

    # The "width" for the RNN is the sequence length output by the CNN.
    # This needs to be calculated based on the model architecture.
    # For the model in model_definition.py, W_out = W_in/4 + 3 (approx)
    # Let's assume the model's forward pass handles this and we just need to pass the widths.
    # A simpler approach for CTC is to pass the output sequence length from the model.
    # The CTC loss function in PyTorch needs the *input* lengths for the RNN part.
    # So, we calculate the sequence length after the CNN.
    # This is a bit tricky as it couples the dataset to the model architecture.
    # A common approach is to have the model compute this or pass a downsampling factor.

    # Let's assume a downsampling factor of 4 from the CNN.
    # This is a simplification.
    cnn_downsample_factor = 4 # This must match your model's CNN!
    image_widths = torch.IntTensor([img.shape[2] // cnn_downsample_factor for img in images])

    # --- Handle Labels ---
    label_lengths = torch.IntTensor([len(label) for label in labels])

    # Pad labels to the length of the longest label in the batch
    # The padding value should be something ignored by the loss function.
    # For CTC, the target padding value doesn't matter as much as providing the correct lengths.
    # We can use 0, but our label encoding starts at 1. So 0 is fine as a padding value.
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    return images_stacked, labels_padded, image_widths, label_lengths


if __name__ == '__main__':
    print("\n--- Testing OCR Dataset and DataLoader (Placeholders) ---")

    # Create dummy files and directories for testing
    test_dir = "temp_ocr_dataset_test"
    img_dir = os.path.join(test_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Dummy annotation file (CSV)
    annotations_path = os.path.join(test_dir, "annotations.csv")
    dummy_annotations = {
        'filename': ['img1.png', 'img2.png', 'img3.jpg', 'non_existent.png'],
        'text': ['hello', 'world', 'test', 'bad_image']
    }
    pd.DataFrame(dummy_annotations).to_csv(annotations_path, index=False)

    # Dummy image files
    import cv2
    cv2.imwrite(os.path.join(img_dir, "img1.png"), np.zeros((32, 100, 3), dtype=np.uint8))
    cv2.imwrite(os.path.join(img_dir, "img2.png"), np.zeros((32, 120, 3), dtype=np.uint8))
    cv2.imwrite(os.path.join(img_dir, "img3.jpg"), np.zeros((32, 80, 3), dtype=np.uint8))

    # Dummy character list
    char_list_path = os.path.join(test_dir, "chars.txt")
    with open(char_list_path, "w", encoding="utf-8") as f:
        f.write("h\ne\nl\no\nw\nr\nd\nt\ns\n")

    print("\n--- Initializing OCRDataset ---")
    try:
        ocr_dataset = OCRDataset(
            annotations_file=annotations_path,
            img_dir=img_dir,
            char_list_path=char_list_path,
            image_height=32,
            image_width=128 # Fixed width for this test
        )
        print(f"Dataset size: {len(ocr_dataset)}")

        # Test __getitem__
        print("\n--- Testing dataset __getitem__ ---")
        # Note: The 4th item in annotations is a non-existent image, so __getitem__ should handle it.
        # Due to the recursive call on error, it might skip it and return the first item again.
        # len() will still be 4, but accessing index 3 will trigger the error handling.
        image_sample, label_sample = ocr_dataset[0]
        print(f"Sample 0: Image shape={image_sample.shape}, Label={label_sample}")
        print(f"Label decoded (dummy): {''.join([ocr_dataset.int_to_char[i.item()] for i in label_sample])}")

        # Test DataLoader with collate_fn
        print("\n--- Testing DataLoader with ocr_collate_fn ---")
        # The dataset will have 3 valid items.
        data_loader = DataLoader(ocr_dataset, batch_size=2, shuffle=False, collate_fn=ocr_collate_fn)

        batch = next(iter(data_loader))
        images_b, labels_b, img_widths_b, label_lengths_b = batch

        print(f"Batch images shape: {images_b.shape}") # (B, C, H, W)
        print(f"Batch labels shape: {labels_b.shape}") # (B, MaxLabelLen)
        print(f"Batch image widths (for RNN): {img_widths_b}")
        print(f"Batch label lengths: {label_lengths_b}")

        if images_b.shape[0] == 2 and labels_b.shape[0] == 2:
            print("DataLoader test conceptually PASSED.")
        else:
            print(f"DataLoader test FAILED. Expected batch size 2, got {images_b.shape[0]}")

    except Exception as e:
        print(f"An error occurred during dataset/dataloader test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy files
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")

    print("\n--- Dataset Script Finished ---")
