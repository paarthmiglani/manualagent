# ocr/image_processor.py
import cv2
import numpy as np

def load_image(image_path_or_data):
    """
    Loads an image from a file path or directly if it's already image data.
    Args:
        image_path_or_data (str or np.ndarray): Path to the image file or image data.
    Returns:
        np.ndarray: The loaded image as a NumPy array (BGR format by default with OpenCV).
                    Returns None if loading fails.
    """
    if isinstance(image_path_or_data, str):
        try:
            image = cv2.imread(image_path_or_data)
            if image is None:
                print(f"Error: Unable to load image from path: {image_path_or_data}")
                return None
            print(f"Image loaded from path: {image_path_or_data}")
            return image
        except Exception as e:
            print(f"Exception while loading image {image_path_or_data}: {e}")
            return None
    elif isinstance(image_path_or_data, np.ndarray):
        print("Image data provided directly.")
        return image_path_or_data
    else:
        print("Error: Invalid image input type. Must be a file path (str) or NumPy array.")
        return None

def preprocess_image_for_ocr(image, target_size=None, grayscale=True, normalize=True):
    """
    Preprocesses an image for OCR.
    Steps typically include:
    - Resizing (optional)
    - Conversion to grayscale
    - Normalization
    - Noise reduction (e.g., Gaussian blur)
    - Binarization (e.g., Otsu's thresholding) - might be model-specific

    Args:
        image (np.ndarray): Input image (BGR or grayscale).
        target_size (tuple, optional): (width, height) to resize the image. Defaults to None.
        grayscale (bool): Whether to convert the image to grayscale. Defaults to True.
        normalize (bool): Whether to normalize pixel values (e.g., to [0, 1] or mean/std). Defaults to True.

    Returns:
        np.ndarray: The preprocessed image.
    """
    if image is None:
        print("Error: Cannot preprocess a None image.")
        return None

    processed_image = image.copy()

    if target_size:
        processed_image = cv2.resize(processed_image, target_size, interpolation=cv2.INTER_AREA)
        print(f"Image resized to: {target_size}")

    if grayscale:
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3: # Check if it's a color image
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            print("Image converted to grayscale.")
        elif len(processed_image.shape) == 2:
            print("Image is already grayscale.")
        else:
            print("Warning: Image is not in expected BGR or Grayscale format for grayscale conversion.")


    # Noise reduction - simple Gaussian blur as an example
    # processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)
    # print("Applied Gaussian blur (example).")

    # Binarization - example using Otsu's thresholding
    # This step is often highly dependent on the OCR model being used.
    # Some models prefer grayscale, others binary.
    # if grayscale: # Typically apply thresholding on grayscale images
    #     _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     print("Applied Otsu's thresholding (example).")

    if normalize:
        # Simple normalization to [0, 1] range if image is 8-bit
        if processed_image.dtype == np.uint8:
            processed_image = processed_image.astype(np.float32) / 255.0
            print("Image normalized to [0, 1] range.")
        # More complex normalization (e.g., Z-score) could be added here based on model requirements

    print("Image preprocessing complete.")
    return processed_image


def detect_text_regions(image, detection_model=None):
    """
    Detects text regions in an image using a custom-trained text detection model.
    This is a placeholder for integrating a model like EAST, CRAFT, or a custom solution.

    Args:
        image (np.ndarray): Preprocessed image suitable for the detection model.
        detection_model: The loaded text detection model.

    Returns:
        list: A list of bounding boxes (e.g., [x, y, w, h] or [x1, y1, x2, y2])
              for each detected text region.
              Each element could also be a dictionary with 'bbox' and 'confidence'.
    """
    if detection_model is None:
        print("Placeholder: No text detection model provided. Returning dummy regions.")
        # Dummy bounding boxes for illustration
        # Assuming image shape is (height, width) or (height, width, channels)
        h, w = image.shape[:2]
        dummy_regions = [
            {'bbox': [int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.2)], 'confidence': 0.95}, # x, y, x+width, y+height format
            {'bbox': [int(w*0.2), int(h*0.4), int(w*0.7), int(h*0.15)], 'confidence': 0.90}
        ]
        return dummy_regions

    print("Detecting text regions using the provided model (placeholder)...")
    # Actual implementation would involve:
    # 1. Formatting the image for the model input
    # 2. Performing inference with the detection_model
    # 3. Post-processing the model output to get bounding boxes
    # text_regions = detection_model.predict(image)
    # return postprocess_detection_output(text_regions)
    return []


if __name__ == '__main__':
    # Create a dummy image for testing (e.g., a 300x400 color image)
    dummy_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    cv2.imwrite("dummy_test_image.png", dummy_img)
    print("Dummy test image 'dummy_test_image.png' created.")

    # 1. Load image
    img = load_image("dummy_test_image.png")

    if img is not None:
        # 2. Preprocess image
        preprocessed_img = preprocess_image_for_ocr(img, target_size=(600, 800), grayscale=True, normalize=True)
        if preprocessed_img is not None:
            print(f"Preprocessed image shape: {preprocessed_img.shape}, dtype: {preprocessed_img.dtype}")
            # cv2.imwrite("dummy_preprocessed_image.png", (preprocessed_img * 255).astype(np.uint8) if preprocessed_img.dtype==np.float32 else preprocessed_img)


            # 3. Detect text regions (using placeholder)
            regions = detect_text_regions(preprocessed_img, detection_model=None)
            print("\nDetected Text Regions (Placeholder):")
            for region in regions:
                print(f"  Bounding Box: {region['bbox']}, Confidence: {region.get('confidence', 'N/A')}")

            # Example of cropping a region (requires original or appropriately scaled image)
            # if regions and img is not None:
            #     box = regions[0]['bbox']
            #     # Ensure box coordinates are valid for the image they are applied to
            #     # (e.g., if preprocessing resized, boxes need to be scaled back or applied to preprocessed_img)
            #     # For this example, let's assume preprocessed_img is the target for cropping
            #     # and box coordinates are relative to it.
            #     # Coordinates might need conversion if they are not in x_min, y_min, x_max, y_max format.
            #     # Assuming box is [x1, y1, x2, y2] relative to preprocessed_img
            #     x1, y1, x2, y2 = box
            #     # Ensure coordinates are integers for slicing
            #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #     # Ensure coordinates are within image bounds
            #     h_proc, w_proc = preprocessed_img.shape[:2]
            #     x1, y1 = max(0, x1), max(0, y1)
            #     x2, y2 = min(w_proc, x2), min(h_proc, y2)

            #     if y2 > y1 and x2 > x1 : # Check for valid crop dimensions
            #         cropped_region = preprocessed_img[y1:y2, x1:x2]
            #         print(f"\nCropped first region with shape: {cropped_region.shape}")
            #         # cv2.imwrite("dummy_cropped_region.png", (cropped_region * 255).astype(np.uint8) if cropped_region.dtype==np.float32 else cropped_region)
            #     else:
            #         print("\nCould not crop region due to invalid dimensions after clamping.")

    print("\nNote: This is a placeholder script. Implement actual model loading and processing.")
