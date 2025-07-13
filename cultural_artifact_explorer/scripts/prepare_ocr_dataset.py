# scripts/prepare_ocr_dataset.py
# Script to automate the preparation of OCR dataset files for training and validation.

import os
import argparse
import pandas as pd
import yaml
from tqdm import tqdm

def find_corresponding_gt_file(image_filename, directory):
    """
    Finds the ground truth .txt file for a given image filename.
    Handles variations like 'img_123.jpg' -> 'gt_img_123.txt'.
    """
    base_name = os.path.splitext(image_filename)[0]
    if base_name.startswith('img_'):
        gt_base_name = 'gt_' + base_name
    else:
        gt_base_name = 'gt_' + base_name
    gt_filename = gt_base_name + ".txt"
    gt_path = os.path.join(directory, gt_filename)
    return gt_path if os.path.exists(gt_path) else None

def process_directory(data_directory):
    """
    Scans a single directory to extract image-text pairs and all unique characters.
    """
    if not os.path.isdir(data_directory):
        print(f"Error: Directory not found at '{data_directory}'")
        return [], set()

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    annotation_data = []
    all_characters = set()

    image_filenames = [f for f in os.listdir(data_directory) if os.path.splitext(f)[1].lower() in image_extensions]

    print(f"Scanning {data_directory}: Found {len(image_filenames)} potential image files.")

    for image_name in tqdm(image_filenames, desc=f"Processing {os.path.basename(data_directory)}"):
        gt_file_path = find_corresponding_gt_file(image_name, data_directory)
        if gt_file_path:
            try:
                with open(gt_file_path, 'r', encoding='utf-8') as f:
                    text_label = f.read().strip()
                if text_label:
                    annotation_data.append({'filename': image_name, 'text': text_label, 'directory': data_directory})
                    all_characters.update(text_label)
                else:
                    print(f"Warning: Annotation file '{gt_file_path}' is empty. Skipping.")
            except Exception as e:
                print(f"Warning: Error reading '{gt_file_path}': {e}. Skipping.")
        else:
            print(f"Warning: No annotation found for image '{image_name}'. Skipping.")

    return annotation_data, all_characters

def update_ocr_config(config_path, train_ann_path, val_ann_path, char_list_path, train_img_dir, val_img_dir):
    """
    Updates the configs/ocr.yaml file with the paths to the generated dataset files.
    """
    if not os.path.exists(config_path):
        print(f"Warning: OCR config file not found at '{config_path}'. A new one will be created.")
        config_data = {}
    else:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}

    if 'model' not in config_data: config_data['model'] = {}
    if 'training' not in config_data: config_data['training'] = {}

    # Use absolute paths to prevent issues with different working directories
    config_data['training']['dataset_path'] = os.path.abspath(train_img_dir)
    config_data['training']['annotations_file'] = os.path.abspath(train_ann_path)
    if val_ann_path:
        config_data['training']['validation_dataset_path'] = os.path.abspath(val_img_dir)
        config_data['training']['validation_annotations_file'] = os.path.abspath(val_ann_path)

    config_data['model']['char_list_path'] = os.path.abspath(char_list_path)

    os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)

    print(f"\nSuccessfully updated '{config_path}' with the new dataset paths.")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare OCR dataset files (annotations.csv, char_list.txt) for training and validation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the directory with training images and annotations.")
    parser.add_argument('--val_dir', type=str, help="(Optional) Path to the directory with validation images and annotations.")
    parser.add_argument('--config_path', type=str, default="configs/ocr.yaml", help="Path to the OCR configuration file to update.")

    args = parser.parse_args()

    print("--- Starting OCR Dataset Preparation ---")

    # Process training directory
    train_data, train_chars = process_directory(args.train_dir)
    if not train_data:
        print("Preparation failed: No training data could be processed.")
        return

    # Process validation directory if provided
    val_data, val_chars = [], set()
    if args.val_dir:
        val_data, val_chars = process_directory(args.val_dir)

    # Combine character sets from both train and val to create a unified vocabulary
    combined_chars = train_chars.union(val_chars)

    # --- Create output files in the project's data/ocr directory ---
    output_dir = os.path.join("data", "ocr")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save training annotations
    train_df = pd.DataFrame(train_data)[['filename', 'text']]
    train_ann_path = os.path.join(output_dir, "train_annotations.csv")
    train_df.to_csv(train_ann_path, index=False)
    print(f"\nTraining annotations file created at: {train_ann_path} ({len(train_df)} entries)")

    # 2. Save validation annotations if they exist
    val_ann_path = None
    if val_data:
        val_df = pd.DataFrame(val_data)[['filename', 'text']]
        val_ann_path = os.path.join(output_dir, "val_annotations.csv")
        val_df.to_csv(val_ann_path, index=False)
        print(f"Validation annotations file created at: {val_ann_path} ({len(val_df)} entries)")

    # 3. Save the combined character list
    sorted_chars = sorted(list(combined_chars))
    char_list_path = os.path.join(output_dir, "char_list.txt")
    with open(char_list_path, 'w', encoding='utf-8') as f:
        for char in sorted_chars:
            f.write(char + '\n')
    print(f"Character list created with {len(sorted_chars)} unique characters at: {char_list_path}")

    # 4. Update the main config file
    update_ocr_config(
        config_path=args.config_path,
        train_ann_path=train_ann_path,
        val_ann_path=val_ann_path,
        char_list_path=char_list_path,
        train_img_dir=args.train_dir,
        val_img_dir=args.val_dir
    )

    print("\n--- Preparation Complete! ---")
    print("You are now ready to run the training script, which will also perform validation if a validation set was provided.")
    print(f"  python -m src.ocr.train --config {args.config_path}")

if __name__ == '__main__':
    main()
