# ocr/visualizer.py
import cv2
import numpy as np

def draw_ocr_results(image, ocr_data, color=(0, 255, 0), thickness=2, font_scale=0.7, font_color=(255, 0, 0)):
    """
    Draws bounding boxes and recognized text onto an image.

    Args:
        image (np.ndarray): The input image (BGR format).
        ocr_data (list): A list of dictionaries, where each dictionary should have
                         at least a 'bbox' key. Optionally, a 'text' key for recognized text.
                         'bbox' can be [x_min, y_min, x_max, y_max] or [x, y, w, h].
                         Example: [{'bbox': [10, 20, 100, 50], 'text': 'Hello'}, ...]
        color (tuple): BGR color for the bounding box. Defaults to green (0, 255, 0).
        thickness (int): Thickness of the bounding box lines. Defaults to 2.
        font_scale (float): Font scale for the recognized text. Defaults to 0.7.
        font_color (tuple): BGR color for the text. Defaults to blue (255, 0, 0).

    Returns:
        np.ndarray: The image with OCR results drawn on it.
    """
    if image is None:
        print("Error: Input image for visualization is None.")
        return None

    output_image = image.copy()

    for item in ocr_data:
        bbox = item.get('bbox')
        text = item.get('text')
        # confidence = item.get('confidence') # Could be used to color-code boxes or text

        if not bbox:
            print(f"Warning: Skipping item due to missing 'bbox': {item}")
            continue

        # Assuming bbox can be [x_min, y_min, x_max, y_max] or [x, y, w, h]
        # Convert to [x_min, y_min, x_max, y_max] if necessary
        if len(bbox) == 4:
            # Check if it's [x,y,w,h] by seeing if w or h are larger than typical coordinates
            # This heuristic is not foolproof. A fixed format is better.
            # For now, let's assume it's [x_min, y_min, x_max, y_max] or that
            # the user ensures the correct format.
            # If it was [x,y,w,h]: pt1 = (bbox[0], bbox[1]); pt2 = (bbox[0]+bbox[2], bbox[1]+bbox[3])
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))

            # Draw bounding box
            cv2.rectangle(output_image, pt1, pt2, color, thickness)

            # Put text if available
            if text:
                # Position text slightly above the bounding box or inside at the top
                text_origin = (pt1[0], pt1[1] - 10 if pt1[1] - 10 > 10 else pt1[1] + 15)
                cv2.putText(output_image, str(text), text_origin,
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
        else:
            print(f"Warning: Skipping item due to invalid 'bbox' format (expected 4 values): {bbox}")


    print("OCR results drawn on image.")
    return output_image

if __name__ == '__main__':
    # Create a dummy image (e.g., a 400x600 color image)
    dummy_height, dummy_width = 400, 600
    dummy_image = np.full((dummy_height, dummy_width, 3), (200, 200, 200), dtype=np.uint8) # Light gray background
    cv2.imwrite("dummy_visualization_input.png", dummy_image)
    print("Dummy input image 'dummy_visualization_input.png' created.")

    # Dummy OCR data
    sample_ocr_data = [
        {'bbox': [50, 50, 250, 100], 'text': 'Devanagari: देवनागरी', 'confidence': 0.92},
        {'bbox': [300, 120, 550, 180], 'text': 'Tamil: தமிழ்', 'confidence': 0.88},
        {'bbox': [50, 200, 550, 350], 'text': 'English: Cultural Artifacts', 'confidence': 0.95},
        {'bbox': [10, 10, 40, 40]} # Box without text
    ]

    # Load the image (as if it's an external image)
    image_to_visualize = cv2.imread("dummy_visualization_input.png")

    if image_to_visualize is not None:
        # Draw OCR results
        visualized_image = draw_ocr_results(
            image_to_visualize,
            sample_ocr_data,
            color=(0, 128, 0),      # Darker green for boxes
            thickness=1,
            font_scale=0.6,
            font_color=(128, 0, 0)  # Darker blue for text
        )

        if visualized_image is not None:
            # Save or display the image
            cv2.imwrite("dummy_ocr_visualization_output.png", visualized_image)
            print("Saved 'dummy_ocr_visualization_output.png' with OCR results.")

            # To display (optional, may not work in all environments)
            # cv2.imshow("OCR Visualization", visualized_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    else:
        print("Failed to load the dummy input image for visualization.")

    print("\nNote: This script provides basic visualization. More advanced features "
          "like handling overlapping text or different font supports for Indic scripts "
          "would require more complex implementation (e.g., using Pillow with specific fonts).")
