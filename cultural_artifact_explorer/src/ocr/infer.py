# src/ocr/infer.py
# Script for OCR model inference.

import yaml
import argparse
import os
import torch
import numpy as np
import cv2

# Import model, dataset, and utilities from our source files
from .model_definition import CRNN
from .utils import preprocess_image_for_ocr, load_char_list
from .postprocess import ctc_decode_predictions

class OCRInferencer:
    def __init__(self, config_path):
        """Initializes the inferencer with configuration and a trained model."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config.get('model', {})
        self.preprocess_config = self.config.get('preprocessing', {})
        self.postprocess_config = self.config.get('postprocessing', {})
        self.infer_config = self.config.get('inference', {})

        print(f"OCRInferencer initialized with config: {config_path}")
        self._setup_device()
        self._load_char_list()
        self._load_model()

    def _setup_device(self):
        """Sets up the device for inference (CPU or GPU)."""
        device_str = self.infer_config.get('device', 'cpu')
        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

    def _load_char_list(self):
        """Loads the character list used by the model."""
        char_list_path = self.model_config.get('char_list_path')
        if not char_list_path:
            raise ValueError("Character list path ('char_list_path') not specified in OCR config.")
        self.char_list = load_char_list(char_list_path)
        print(f"Character list loaded. Vocab size: {len(self.char_list)}.")

    def _load_model(self):
        """Loads the trained CRNN model weights."""
        print("Loading OCR model for inference...")
        model_weights_path = self.model_config.get('weights_path')
        if not model_weights_path or not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weights_path not specified or not found at '{model_weights_path}' in OCR config.")

        num_classes = len(self.char_list) + 1  # +1 for blank

        self.model = CRNN(
            img_channels=self.model_config.get('input_channels', 1),
            num_classes=num_classes,
            rnn_hidden_size=self.model_config.get('rnn_hidden_size', 256),
            rnn_num_layers=self.model_config.get('num_rnn_layers', 2)
        ).to(self.device)

        try:
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            print("This can happen if the model architecture in model_definition.py does not match the saved weights.")
            raise

        self.model.eval() # Set model to evaluation mode
        print(f"OCR model loaded successfully from {model_weights_path}.")

    def predict(self, image_path):
        """
        Performs OCR on a single image.
        Args:
            image_path (str): Path to the image file.
        Returns:
            str: Recognized text.
            float: Confidence score of the recognition.
        """
        print(f"Performing OCR prediction for: {image_path}...")

        # 1. Preprocess the image
        try:
            image_tensor = preprocess_image_for_ocr(
                image_path,
                target_size=(self.model_config.get('input_width', 128), self.model_config.get('input_height', 32)),
                binarize=self.preprocess_config.get('binarize', False)
            )
            # Add batch dimension and move to device
            image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return "Error in preprocessing", 0.0

        # 2. Perform model inference
        with torch.no_grad():
            log_probs = self.model(image_tensor) # Shape: (SeqLen, Batch=1, NumClasses)

        # 3. Decode the output
        # Squeeze batch dimension and convert to numpy array on CPU
        raw_preds_np = log_probs.squeeze(1).cpu().numpy()

        # We need probabilities for confidence, not log_probs
        # preds_probs = np.exp(raw_preds_np)

        # Use the postprocessing function to decode
        recognized_text, confidence = ctc_decode_predictions(
            raw_preds_np, # The function expects log_probs or logits, let's pass log_probs
            self.char_list,
            beam_search=self.postprocess_config.get('beam_search', False),
            beam_width=self.postprocess_config.get('beam_width', 5),
            blank_idx=0
        )

        print(f"  Decoded Text: '{recognized_text}', Confidence: {confidence:.4f}")
        return recognized_text, confidence

def main():
    parser = argparse.ArgumentParser(description="Run OCR inference on an image.")
    parser.add_argument('--image', type=str, required=True, help="Path to the image file.")
    parser.add_argument('--config', type=str, default="configs/ocr.yaml",
                        help="Path to the OCR configuration YAML file.")

    args = parser.parse_args()

    # --- Dummy File Creation for Placeholder Run ---
    if not os.path.exists(args.config) or "temp_chars.txt" in open(args.config).read():
        print(f"Warning: Config file not found or is a dummy. Creating dummy files for inference test.")
        # Create dummy config
        dummy_model_path = "temp_dummy_model.pth"
        dummy_char_path = "temp_chars.txt"
        dummy_cfg = {
            'model': {
                'weights_path': dummy_model_path,
                'char_list_path': dummy_char_path,
                'input_height': 32, 'input_width': 128, 'input_channels': 1,
                'rnn_hidden_size': 256, 'rnn_num_layers': 2
            },
            'preprocessing': {'binarize': False},
            'postprocessing': {'beam_search': False},
            'inference': {'device': 'cpu'}
        }
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f: yaml.dump(dummy_cfg, f)

        # Create dummy char list
        with open(dummy_char_path, 'w') as f: f.write('a\nb\nc\n')

        # Create dummy model
        char_list = load_char_list(dummy_char_path)
        num_classes = len(char_list) + 1
        dummy_model = CRNN(img_channels=1, num_classes=num_classes)
        torch.save(dummy_model.state_dict(), dummy_model_path)

        # Create dummy image
        if not os.path.exists(args.image):
            os.makedirs(os.path.dirname(args.image) or '.', exist_ok=True)
            cv2.imwrite(args.image, np.zeros((32, 128, 3), dtype=np.uint8))
            print(f"Created dummy image at {args.image}")

    # --- Main Execution ---
    try:
        inferencer = OCRInferencer(config_path=args.config)
        text, confidence = inferencer.predict(image_path=args.image)

        print("\n--- OCR Inference Result ---")
        print(f"  Input Image: {args.image}")
        print(f"  Recognized Text: '{text}'")
        print(f"  Confidence: {confidence:.4f}")
        print("--------------------------")

    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # --- Clean up dummy files ---
        with open(args.config, 'r') as f:
            content = f.read()
        if "temp_chars.txt" in content:
            print("\nCleaning up dummy files created for the test run...")
            config_data = yaml.safe_load(content)
            os.remove(config_data['model']['weights_path'])
            os.remove(config_data['model']['char_list_path'])
            # os.remove(args.config)
            if not os.path.dirname(args.image): # if image is in root
                 if os.path.exists(args.image): os.remove(args.image)
            # Be careful cleaning up user-provided paths if they weren't dummy
            # This logic assumes if config was dummy, image was too.

if __name__ == '__main__':
    # To run, you need a trained model (.pth), a char_list.txt, and a config.yaml pointing to them.
    # The main function here creates dummy versions of these to allow a dry run.
    # Example: python src/ocr/infer.py --image my_test_image.png --config my_ocr_config.yaml
    print("Executing src.ocr.infer (with implemented inference logic)...")
    # To run the placeholder test, provide a dummy image path. e.g., --image temp.png
    # The script will create it if it doesn't exist (if it also creates the dummy config).
    main()
