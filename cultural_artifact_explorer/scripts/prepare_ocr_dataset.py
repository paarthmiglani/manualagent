# scripts/prepare_ocr_dataset.py
# Script to automate the preparation of OCR dataset files.

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
    base_name = os.path.splitext(image_filename)[0] # e.g., "img_123"

    # Construct the expected ground truth filename
    # Assumes the pattern is replacing 'img_' with 'gt_img_'
    if base_name.startswith('img_'):
        gt_base_name = 'gt_' + base_name
    else:
        # Fallback for other naming conventions if needed
        gt_base_name = 'gt_' + base_name

    gt_filename = gt_base_name + ".txt"
    gt_path = os.path.join(directory, gt_filename)

    if os.path.exists(gt_path):
        return gt_path
    return None

def create_ocr_dataset_files(data_directory):
    """
    Scans a directory for images and ground truth text files, then creates
    a consolidated annotations.csv and a character list file.

    Args:
        data_directory (str): The path to the directory containing both
                              image files (e.g., .jpg, .png) and ground truth
                              text files (e.g., gt_img_xxxx.txt).
    """
    print(f"Scanning directory: {data_directory}")

    if not os.path.isdir(data_directory):
        print(f"Error: Directory not found at '{data_directory}'")
        return None, None

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    annotation_data = []
    all_characters = set()

    # Find all image files and their corresponding annotation files
    filenames_in_dir = os.listdir(data_directory)
    image_filenames = [f for f in filenames_in_dir if os.path.splitext(f)[1].lower() in image_extensions]

    print(f"Found {len(image_filenames)} potential image files. Looking for corresponding annotations...")

    for image_name in tqdm(image_filenames, desc="Processing files"):
        gt_file_path = find_corresponding_gt_file(image_name, data_directory)

        if gt_file_path:
            try:
                with open(gt_file_path, 'r', encoding='utf-8') as f:
                    text_label = f.read().strip()

                if text_label: # Only add if there is text content
                    annotation_data.append({'filename': image_name, 'text': text_label})
                    # Update the character set
                    all_characters.update(text_label)
                else:
                    print(f"Warning: Annotation file '{gt_file_path}' is empty. Skipping.")

            except Exception as e:
                print(f"Warning: Error reading or processing '{gt_file_path}': {e}. Skipping.")
        else:
            print(f"Warning: No corresponding annotation file found for image '{image_name}'. Skipping.")

    if not annotation_data:
        print("Error: No valid image-annotation pairs were found. Cannot create dataset files.")
        return None, None

    print(f"\nSuccessfully processed {len(annotation_data)} image-annotation pairs.")

    # --- Create the output files in the project's data/ocr directory ---
    output_dir = os.path.join("data", "ocr")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create and save the annotations.csv file
    annotations_df = pd.DataFrame(annotation_data)
    annotations_csv_path = os.path.join(output_dir, "annotations.csv")
    annotations_df.to_csv(annotations_csv_path, index=False)
    print(f"Annotations file created at: {annotations_csv_path}")

    # 2. Create and save the char_list.txt file
    sorted_chars = sorted(list(all_characters))
    char_list_path = os.path.join(output_dir, "char_list.txt")
    with open(char_list_path, 'w', encoding='utf-8') as f:
        for char in sorted_chars:
            f.write(char + '\n')
    print(f"Character list created with {len(sorted_chars)} unique characters at: {char_list_path}")

    return annotations_csv_path, char_list_path

def update_ocr_config(config_path, dataset_path, annotations_path, char_list_path):
    """
    Updates the configs/ocr.yaml file with the paths to the generated dataset files.
    """
    if not os.path.exists(config_path):
        print(f"Warning: OCR config file not found at '{config_path}'. A new one will be created.")
        config_data = {}
    else:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}

    # Ensure nested keys exist
    if 'model' not in config_data: config_data['model'] = {}
    if 'training' not in config_data: config_data['training'] = {}

    # Get relative paths for portability
    # Assumes script is run from project root, so paths should be relative from there.
    rel_dataset_path = os.path.relpath(dataset_path, start=os.getcwd())
    rel_annotations_path = os.path.relpath(annotations_path, start=os.getcwd())
    rel_char_list_path = os.path.relpath(char_list_path, start=os.getcwd())

    # Update the paths
    config_data['training']['dataset_path'] = rel_dataset_path
    config_data['training']['annotations_file'] = rel_annotations_path
    config_data['model']['char_list_path'] = rel_char_list_path

    # Write the updated config back to the file
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)

    print(f"\nSuccessfully updated '{config_path}' with the new dataset paths.")
    print("  training.dataset_path ->", rel_dataset_path)
    print("  training.annotations_file ->", rel_annotations_path)
    print("  model.char_list_path ->", rel_char_list_path)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare OCR dataset files (annotations.csv, char_list.txt) and update OCR config.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help="Path to the directory containing both image files (e.g., *.jpg) and annotation files (e.g., gt_*.txt)."
    )
    parser.add_argument(
        '--config_path', type=str, default="configs/ocr.yaml",
        help="Path to the OCR configuration file to update."
    )

    args = parser.parse_args()

    print("--- Starting OCR Dataset Preparation ---")

    # 1. Create the dataset files
    ann_path, char_path = create_ocr_dataset_files(args.data_dir)

    # 2. Update the config file if files were created successfully
    if ann_path and char_path:
        update_ocr_config(
            config_path=args.config_path,
            dataset_path=args.data_dir, # The dataset_path in config should point to the images
            annotations_path=ann_path,
            char_list_path=char_path
        )
        print("\n--- Preparation Complete! ---")
        print("You are now ready to run the training script:")
        print(f"  python src/ocr/train.py --config {args.config_path}")
    else:
        print("\n--- Preparation Failed. Please check the errors above. ---")


if __name__ == '__main__':
    # Example usage:
    # From the project root (`cultural_artifact_explorer`):
    # python scripts/prepare_ocr_dataset.py --data_dir /path/to/your/images_and_annotations
    main()
