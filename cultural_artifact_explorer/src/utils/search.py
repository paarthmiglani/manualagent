import os
import pandas as pd

def find_translation_files(dir_path):
    """
    Finds all JSON and CSV files in a directory that are suitable for translation tasks.
    """
    supported_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(('.json', '.csv')):
                supported_files.append(os.path.join(root, file))
    return supported_files

def read_translation_data(file_path):
    """
    Reads a translation data file (JSON or CSV) and returns a list of dictionaries.
    Each dictionary should have 'source_text' and 'target_text' keys.
    """
    if file_path.endswith('.json'):
        data = pd.read_json(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Only JSON and CSV are supported.")

    # Basic validation to ensure the required columns are present
    if 'source_text' not in data.columns or 'target_text' not in data.columns:
        raise ValueError("Translation files must contain 'source_text' and 'target_text' columns.")

    return data.to_dict('records')
