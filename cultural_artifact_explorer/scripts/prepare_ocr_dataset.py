# scripts/prepare_ocr_dataset.py

# Script to automate the preparation of OCR dataset files for training and validation,
# now with support for multiple data directories to create a unified multilingual dataset.
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
def process_directories(data_directories):
    """
    Scans a list of directories to extract image-text pairs and all unique characters.
    """
    all_annotation_data = []
    all_characters = set()

    for data_dir in data_directories:
        if not os.path.isdir(data_dir):
            print(f"Warning: Directory not found at '{data_dir}'. Skipping.")
            continue

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_filenames = [f for f in os.listdir(data_dir) if os.path.splitext(f)[1].lower() in image_extensions]

        print(f"Scanning {data_dir}: Found {len(image_filenames)} potential image files.")

        for image_name in tqdm(image_filenames, desc=f"Processing {os.path.basename(data_dir)}"):
            gt_file_path = find_corresponding_gt_file(image_name, data_dir)
            if gt_file_path:
                try:
                    with open(gt_file_path, 'r', encoding='utf-8') as f:
                        text_label = f.read().strip()
                    if text_label:
                        # We need to store the absolute path to the image now
                        abs_image_path = os.path.abspath(os.path.join(data_dir, image_name))
                        all_annotation_data.append({'filepath': abs_image_path, 'text': text_label})
                        all_characters.update(text_label)
                    else:
                        print(f"Warning: Annotation file '{gt_file_path}' is empty. Skipping.")
                except Exception as e:
                    print(f"Warning: Error reading '{gt_file_path}': {e}. Skipping.")
            else:
                print(f"Warning: No annotation found for image '{image_name}'. Skipping.")

    return all_annotation_data, all_characters

def update_ocr_config(config_path, train_ann_path, val_ann_path, char_list_path):
    """
    Updates the configs/ocr.yaml file with the paths to the generated dataset files.
    The dataset_path fields are now removed as the annotation files will contain absolute paths.
    """
    if not os.path.exists(config_path):
        print(f"Warning: OCR config file not found at '{config_path}'. A new one will be created.")
        config_data = {}
    else:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}

    if 'model' not in config_data: config_data['model'] = {}
    if 'training' not in config_data: config_data['training'] = {}


    # Remove old dataset_path keys and set new annotation file paths
    config_data['training'].pop('dataset_path', None)
    config_data['training'].pop('validation_dataset_path', None)

    config_data['training']['annotations_file'] = os.path.abspath(train_ann_path)
    if val_ann_path:
        config_data['training']['validation_annotations_file'] = os.path.abspath(val_ann_path)

    config_data['model']['char_list_path'] = os.path.abspath(char_list_path)

    os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)

    print(f"\nSuccessfully updated '{config_path}' with the new dataset paths.")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare multilingual OCR dataset files and update config.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--train_dirs', nargs='+', required=True, help="Space-separated list of directories for training data.")
    parser.add_argument('--val_dirs', nargs='+', help="(Optional) Space-separated list of directories for validation data.")
    parser.add_argument('--config_path', type=str, default="configs/ocr.yaml", help="Path to the OCR configuration file.")

    args = parser.parse_args()

    print("--- Starting Multilingual OCR Dataset Preparation ---")

    # Process training directories
    train_data, train_chars = process_directories(args.train_dirs)
    if not train_data:
        print("Preparation failed: No training data could be processed.")
        return
    # Process validation directories if provided
    val_data, val_chars = [], set()
    if args.val_dirs:
        val_data, val_chars = process_directories(args.val_dirs)

    combined_chars = train_chars.union(val_chars)

    output_dir = os.path.join("data", "ocr")
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    train_ann_path = os.path.join(output_dir, "train_annotations.csv")
    train_df.to_csv(train_ann_path, index=False)
    print(f"\nTraining annotations file created at: {train_ann_path} ({len(train_df)} entries)")

    val_ann_path = None
    if val_data:
        val_df = pd.DataFrame(val_data)
        val_ann_path = os.path.join(output_dir, "val_annotations.csv")
        val_df.to_csv(val_ann_path, index=False)
        print(f"Validation annotations file created at: {val_ann_path} ({len(val_df)} entries)")
    sorted_chars = sorted(list(combined_chars))
    char_list_path = os.path.join(output_dir, "char_list.txt")
    with open(char_list_path, 'w', encoding='utf-8') as f:
        for char in sorted_chars:
            f.write(char + '\n')

    print(f"Unified character list created with {len(sorted_chars)} unique characters at: {char_list_path}")
    update_ocr_config(
        config_path=args.config_path,
        train_ann_path=train_ann_path,
        val_ann_path=val_ann_path,
        char_list_path=char_list_path,
    )

    print("\n--- Preparation Complete! ---")

if __name__ == '__main__':
    main()
