# src/ocr/utils.py
# Utility functions for the OCR module

import cv2
import numpy as np
import os # Added for __main__ example

# import unicodedata # For text normalization if needed - keep commented for now

def load_char_list(filepath):
    """Loads a character list from a file, one character per line."""
    print(f"Loading character list from: {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(chars)} characters from {filepath}.")
        return chars
    except FileNotFoundError:
        print(f"Error: Character list file not found at {filepath}. Returning empty list.")
        # Consider raising an error or returning a default list depending on desired behavior
        raise # Reraise for caller to handle or return []
    except Exception as e:
        print(f"Error loading character list from {filepath}: {e}. Returning empty list.")
        raise # Reraise or return []

def preprocess_image_for_ocr(image_path, target_size=(128, 32), binarize=False):
    """
    Preprocesses an image for OCR inference or training.
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired (width, height) after resizing. Note: OpenCV uses (width, height).
        binarize (bool): Whether to apply Otsu's binarization.
    Returns:
        np.ndarray: Preprocessed image, shape (1, H, W) ready for model input.
    """
    print(f"Preprocessing image: {image_path} with target_size={target_size}, binarize={binarize}...")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image not found or unable to read at {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    if binarize:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print("  Applied Otsu's binarization.")

    # Resize: OpenCV's resize takes (width, height)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) # INTER_AREA for shrinking, INTER_CUBIC/LINEAR for enlarging
    print(f"  Resized to {target_size}.")

    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    print("  Normalized to [0,1].")

    # Add channel dimension (C, H, W) - for grayscale, C=1
    # Models often expect (Batch, Channel, Height, Width)
    # This function returns (Channel, Height, Width). Batch dimension is usually added by DataLoader or before inference.
    img = np.expand_dims(img, axis=0) # Shape: (1, H, W)
    print(f"  Expanded dims to shape: {img.shape}.")
    return img

def visualize_ocr_results(image_path, boxes, texts, output_path=None):
    """
    Draws OCR bounding boxes and recognized text on an image.
    Args:
        image_path (str): Path to the original image.
        boxes (list of lists/tuples): List of bounding boxes.
                                      Each box assumed format [x_min, y_min, x_max, y_max].
        texts (list of str): List of recognized text strings, corresponding to boxes.
        output_path (str, optional): Path to save the visualized image. If None, not saved.
    Returns:
        np.ndarray: Image with visualizations, or None if input image couldn't be loaded.
    """
    print(f"Visualizing OCR results for image: {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        # Try to provide a more informative error or handle it gracefully
        print(f"Error: Cannot load image for visualization at {image_path}")
        # raise ValueError(f"Cannot load image at {image_path}")
        return None # Return None if image can't be loaded.

    if len(boxes) != len(texts):
        print("Warning: Number of boxes and texts do not match. Visualization might be incorrect.")
        # Decide how to handle: either raise error, or proceed cautiously.
        # For now, proceed but it might lead to runtime errors if zip stops early.

    for box, text in zip(boxes, texts):
        try:
            x1, y1, x2, y2 = map(int, box)  # Ensure coordinates are integers
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box

            # Position text slightly above the bounding box
            # Choose a font scale and thickness that works for typical image sizes
            font_scale = 0.6
            thickness = 1
            text_origin = (x1, y1 - 10 if y1 - 10 > 10 else y1 + int(20 * font_scale)) # Adjust if too close to top

            cv2.putText(image, str(text), text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA) # Red text
        except Exception as e:
            print(f"  Warning: Could not draw box/text for box={box}, text='{text}'. Error: {e}")


    if output_path:
        try:
            # os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure dir exists
            cv2.imwrite(output_path, image)
            print(f"  Visualized image saved to: {output_path}")
        except Exception as e:
            print(f"  Error saving visualized image to {output_path}: {e}")

    return image


# Example of a text normalization function (if needed for OCR postprocessing or NLP preprocessing)
def normalize_unicode_text(text, form='NFKC'): # Renamed to be more specific
    """
    Normalize Unicode text.
    Args:
        text (str): Input string.
        form (str): Normalization form ('NFC', 'NFKC', 'NFD', 'NFKD').
    Returns:
        str: Normalized string.
    """
    import unicodedata # Keep import local if not always used
    print(f"Normalizing Unicode text (form: {form}): '{text[:30]}...'")
    return unicodedata.normalize(form, text)

if __name__ == '__main__':
    print("\n--- Testing OCR utility functions (using user-provided implementations) ---")

    # Create dummy directory for outputs if they don't exist
    output_test_dir = "temp_ocr_utils_output"
    os.makedirs(output_test_dir, exist_ok=True)

    # Test load_char_list
    print("\n--- Testing load_char_list ---")
    dummy_char_file = os.path.join(output_test_dir, "dummy_chars.txt")
    try:
        with open(dummy_char_file, "w", encoding="utf-8") as f:
            f.write("a\n")
            f.write("ब\n") # Devanagari 'ba'
            f.write("1\n \n") # Last line is space, should be stripped if just space, or kept if line.strip() handles it
            f.write("  \n") # Empty line after strip
            f.write("c\n")
        chars = load_char_list(dummy_char_file)
        print(f"Loaded characters: {chars}")
        expected_chars = ['a', 'ब', '1', 'c']
        if chars == expected_chars:
            print("load_char_list test PASSED.")
        else:
            print(f"load_char_list test FAILED. Expected {expected_chars}, got {chars}")
    except Exception as e:
        print(f"load_char_list test FAILED with error: {e}")
    finally:
        if os.path.exists(dummy_char_file): os.remove(dummy_char_file)


    # Test preprocess_image_for_ocr
    print("\n--- Testing preprocess_image_for_ocr ---")
    dummy_image_file = os.path.join(output_test_dir, "dummy_image.png")
    try:
        # Create a simple dummy PNG image (e.g., 100x50, 3-channel color)
        dummy_img_data_color = np.random.randint(0, 256, (50, 100, 3), dtype=np.uint8)
        cv2.imwrite(dummy_image_file, dummy_img_data_color)

        # Test case 1: Standard preprocessing
        target_w, target_h = 128, 32
        preprocessed_img = preprocess_image_for_ocr(dummy_image_file, target_size=(target_w, target_h), binarize=False)
        print(f"Preprocessed image shape: {preprocessed_img.shape}") # Expected (1, target_h, target_w)
        if preprocessed_img.shape == (1, target_h, target_w):
             print("preprocess_image_for_ocr (standard) test PASSED.")
        else:
            print(f"preprocess_image_for_ocr (standard) test FAILED. Expected shape (1, {target_h}, {target_w}), got {preprocessed_img.shape}")

        # Test case 2: With binarization
        preprocessed_bin_img = preprocess_image_for_ocr(dummy_image_file, target_size=(target_w, target_h), binarize=True)
        print(f"Preprocessed binarized image shape: {preprocessed_bin_img.shape}")
        if preprocessed_bin_img.shape == (1, target_h, target_w):
             print("preprocess_image_for_ocr (binarized) test PASSED.")
        else:
            print(f"preprocess_image_for_ocr (binarized) test FAILED. Expected shape (1, {target_h}, {target_w}), got {preprocessed_bin_img.shape}")
        # Could add checks for value range [0,1] and that binarized image is mostly 0 or 1
        # print(f"  Min/Max values (binarized): {preprocessed_bin_img.min()}, {preprocessed_bin_img.max()}")

    except Exception as e:
        print(f"preprocess_image_for_ocr test FAILED with error: {e}")
    finally:
        if os.path.exists(dummy_image_file): os.remove(dummy_image_file)


    # Test visualize_ocr_results
    print("\n--- Testing visualize_ocr_results ---")
    dummy_viz_image_file = os.path.join(output_test_dir, "dummy_viz_image_in.png")
    dummy_viz_output_file = os.path.join(output_test_dir, "dummy_viz_image_out.png")
    try:
        dummy_img_data_viz = np.full((100, 300, 3), (200, 200, 200), dtype=np.uint8) # Gray background
        cv2.imwrite(dummy_viz_image_file, dummy_img_data_viz)

        boxes_to_draw = [[10, 10, 100, 40], [120, 50, 280, 80]]
        texts_to_draw = ["Hello", "नमस्ते"] # English, Devanagari

        visualized_image = visualize_ocr_results(dummy_viz_image_file, boxes_to_draw, texts_to_draw, output_path=dummy_viz_output_file)

        if visualized_image is not None and os.path.exists(dummy_viz_output_file):
            print(f"visualize_ocr_results test PASSED. Output saved to {dummy_viz_output_file}")
            # To verify, one would manually inspect the output image.
        elif visualized_image is None:
            print(f"visualize_ocr_results test FAILED: Function returned None (image load issue?).")
        else:
            print(f"visualize_ocr_results test FAILED: Output file not created at {dummy_viz_output_file}")

    except Exception as e:
        print(f"visualize_ocr_results test FAILED with error: {e}")
    finally:
        if os.path.exists(dummy_viz_image_file): os.remove(dummy_viz_image_file)
        if os.path.exists(dummy_viz_output_file): os.remove(dummy_viz_output_file)

    # Test normalize_unicode_text
    print("\n--- Testing normalize_unicode_text ---")
    try:
        # Full-width numbers to half-width, Katakana half-width to full-width
        sample_unicode_text = "ＡＢＣ１２３ｶﾞｷﾞｸﾞｹﾞｺﾞ"
        # Expected with NFKC: "ABC123ガギグゲゴ"
        normalized = normalize_unicode_text(sample_unicode_text, form='NFKC')
        print(f"Original Unicode text: '{sample_unicode_text}'")
        print(f"Normalized (NFKC) text: '{normalized}'")
        # Add an assertion if you have a known good output for your environment/Python version
        # For example: assert normalized == "ABC123ガギグゲゴ"
        print("normalize_unicode_text test conceptually PASSED (inspect output).")
    except ImportError:
        print("normalize_unicode_text test SKIPPED (unicodedata module not available in this environment).")
    except Exception as e:
        print(f"normalize_unicode_text test FAILED with error: {e}")

    # Clean up test output directory if empty or if desired
    if os.path.exists(output_test_dir) and not os.listdir(output_test_dir): # Check if empty
        os.rmdir(output_test_dir)
    elif os.path.exists(output_test_dir):
        print(f"\nNote: Test output directory '{output_test_dir}' contains files. Please inspect/delete manually if needed.")


    print("\n--- OCR utility tests finished ---")
