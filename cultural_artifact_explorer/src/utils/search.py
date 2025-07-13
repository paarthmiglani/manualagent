import os
import pandas as pd
import json

def find_translation_files(dir_path):
    """
    Finds all JSON, JSONL, and CSV files in a directory that are suitable for translation tasks.
    """
    supported_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(('.json', '.jsonl', '.csv')):
                supported_files.append(os.path.join(root, file))
    return supported_files

def read_translation_data(file_path):
    """
    Reads a translation data file (JSON, JSONL, or CSV) and returns a list of dictionaries.
    Each dictionary will be standardized to have 'source_text' and 'target_text' keys.
    """
    if file_path.endswith('.json') or file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Try to parse as JSONL first
        if len(lines) > 1:
            try:
                data = [json.loads(line) for line in lines]
                data = pd.DataFrame(data)
            except json.JSONDecodeError:
                # If that fails, try to parse as a single JSON object
                data = json.loads(''.join(lines))
                if isinstance(data, dict):
                    data = [data]
                data = pd.DataFrame(data)
        else:
            data = json.loads(''.join(lines))
            if isinstance(data, dict):
                data = [data]
            data = pd.DataFrame(data)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Only JSON, JSONL, and CSV are supported.")

    # Define possible column name mappings
    column_mappings = [
        {'source': 'source_text', 'target': 'target_text'},
        {'source': 'native word', 'target': 'english word'},
        {'source': 'Native Word', 'target': 'English Transliteration'},
    ]

    # Find the correct column mapping
    mapping_found = False
    for mapping in column_mappings:
        if mapping['source'] in data.columns and mapping['target'] in data.columns:
            data = data.rename(columns={mapping['source']: 'source_text', mapping['target']: 'target_text'})
            mapping_found = True
            break

    if not mapping_found:
        raise ValueError(f"Could not find required columns in {file_path}")

    return data[['source_text', 'target_text']].to_dict('records')
