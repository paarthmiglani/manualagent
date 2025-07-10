# src/ocr/utils.py
# Utility functions for the OCR module

# import cv2
import numpy as np
# import unicodedata # For text normalization if needed

def load_char_list(file_path):
    """Loads a character list from a file, one character per line."""
    print(f"Loading character list from: {file_path} (placeholder in utils)...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f if line.strip()]
        # It's common to have a <blank> token, often at index 0, for CTC or other schemes.
        # This should be handled consistently by the model and training process.
        # If the file itself doesn't define it, it might be prepended here or in the model.
        # For now, assume the file is comprehensive or model handles special tokens.
        print(f"Loaded {len(chars)} characters from {file_path}.")
        return chars
    except FileNotFoundError:
        print(f"Error: Character list file not found at {file_path}. Returning empty list.")
        return []
    except Exception as e:
        print(f"Error loading character list from {file_path}: {e}. Returning empty list.")
        return []

def preprocess_image_for_ocr(image_data, target_height=64, grayscale=True, invert_colors=False, binarization_threshold=None):
    """
    Preprocesses an image for OCR inference or training.
    Args:
        image_data (np.ndarray): Input image (BGR or Grayscale).
        target_height (int): Desired height after resizing, width is scaled proportionally.
        grayscale (bool): Convert to grayscale if true.
        invert_colors (bool): Invert image colors (e.g., for white text on black bg).
        binarization_threshold (int or str, optional):
            - If int (0-255): Simple threshold.
            - If "otsu": Use Otsu's binarization.
            - If None: No explicit binarization applied here (model might expect grayscale).
    Returns:
        np.ndarray: Preprocessed image.
    """
    print(f"Preprocessing image with target_height={target_height} (placeholder in utils)...")
    # if not isinstance(image_data, np.ndarray):
    #     raise TypeError("image_data must be a NumPy array.")

    # img = image_data.copy()

    # # 1. Convert to Grayscale
    # if grayscale:
    #     if len(img.shape) == 3 and img.shape[2] == 3: # Color image
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     elif len(img.shape) == 3 and img.shape[2] == 1: # Grayscale but with 3rd dim
    #         img = img[:, :, 0]
    # # If already 2D (grayscale), do nothing for this step.

    # # 2. Invert Colors (if needed)
    # if invert_colors:
    #     img = cv2.bitwise_not(img)

    # # 3. Resize to target height, maintaining aspect ratio
    # current_height, current_width = img.shape[:2]
    # if current_height != target_height:
    #     scale_factor = target_height / current_height
    #     target_width = int(current_width * scale_factor)
    #     img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC) # INTER_AREA for shrinking, CUBIC for enlarging

    # # 4. Binarization (optional)
    # if binarization_threshold is not None:
    #     if isinstance(binarization_threshold, str) and binarization_threshold.lower() == 'otsu':
    #         _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     elif isinstance(binarization_threshold, (int, float)):
    #         _, img = cv2.threshold(img, int(binarization_threshold), 255, cv2.THRESH_BINARY)
    #     else:
    #         print(f"Warning: Invalid binarization_threshold value: {binarization_threshold}")

    # # 5. Normalization (e.g., to [0, 1] or [-1, 1] range, often done before feeding to model)
    # # This step is often model-specific and might be part of the model's forward pass or dataset loading.
    # # Example: img = img.astype(np.float32) / 255.0

    # print("Image preprocessing complete (placeholder in utils).")
    # return img

    # Dummy output matching typical OCR model input shape (C, H, W)
    # For placeholder, assume grayscale and fixed width for simplicity
    # In reality, width would vary based on aspect ratio or be padded.
    dummy_width = 200
    num_channels = 1 if grayscale else 3
    return np.random.rand(num_channels, target_height, dummy_width).astype(np.float32)


def visualize_ocr_results(image, ocr_outputs, box_color=(0, 255, 0), text_color=(255, 0, 0), thickness=2):
    """
    Draws OCR bounding boxes and recognized text on an image.
    Args:
        image (np.ndarray): The original image (BGR).
        ocr_outputs (list of dicts): Each dict should have 'bbox' and optionally 'text'.
                                     'bbox' is [x_min, y_min, x_max, y_max].
        box_color (tuple): Color for bounding boxes.
        text_color (tuple): Color for text.
        thickness (int): Line thickness for boxes and text.
    Returns:
        np.ndarray: Image with visualizations.
    """
    print("Visualizing OCR results (placeholder in utils)...")
    # vis_image = image.copy()
    # for output in ocr_outputs:
    #     bbox = output.get('bbox')
    #     text = output.get('text', '')

    #     if bbox and len(bbox) == 4:
    #         x_min, y_min, x_max, y_max = map(int, bbox)
    #         cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), box_color, thickness)
    #         if text:
    #             # Put text slightly above the box
    #             cv2.putText(vis_image, text, (x_min, y_min - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, thickness)
    # return vis_image
    return image # Return original image as placeholder

# Example of a text normalization function (if needed for OCR postprocessing or NLP preprocessing)
def normalize_text(text, form='NFKC'):
    """
    Normalize Unicode text.
    Args:
        text (str): Input string.
        form (str): Normalization form ('NFC', 'NFKC', 'NFD', 'NFKD').
    Returns:
        str: Normalized string.
    """
    # import unicodedata
    # return unicodedata.normalize(form, text)
    return text # Placeholder

if __name__ == '__main__':
    print("Testing OCR utility functions (placeholders)...")

    # Test load_char_list (requires a dummy file)
    dummy_char_list_file = "dummy_chars.txt"
    with open(dummy_char_list_file, "w", encoding="utf-8") as f:
        f.write("a\n")
        f.write("ब\n") # Devanagari 'ba'
        f.write("1\n")
    chars = load_char_list(dummy_char_list_file)
    print(f"Loaded dummy characters: {chars}")
    import os
    os.remove(dummy_char_list_file)

    # Test preprocess_image_for_ocr
    dummy_image_bgr = np.random.randint(0, 256, (100, 300, 3), dtype=np.uint8) # HxWxC
    preprocessed = preprocess_image_for_ocr(dummy_image_bgr, target_height=32, grayscale=True)
    print(f"Preprocessed image shape (dummy): {preprocessed.shape}") # Expected (1, 32, W_scaled_dummy)

    # Test visualize_ocr_results
    dummy_ocr_data = [
        {'bbox': [10, 10, 100, 50], 'text': 'Hello'},
        {'bbox': [120, 60, 250, 100], 'text': 'नमस्ते'} # Devanagari 'Namaste'
    ]
    visualized_img = visualize_ocr_results(dummy_image_bgr, dummy_ocr_data)
    print(f"Visualization (dummy) returned image of shape: {visualized_img.shape}")
    # In a real scenario, you'd save or display visualized_img
    # cv2.imwrite("dummy_ocr_visualization.png", visualized_img)

    # Test normalize_text
    sample_text = "これはﾃｽﾄです。" # Contains half-width Katakana
    normalized = normalize_text(sample_text, form='NFKC')
    print(f"Original text: '{sample_text}', Normalized (NFKC placeholder): '{normalized}'")

    print("OCR utility tests complete (placeholders).")
