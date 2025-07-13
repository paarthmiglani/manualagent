# OCR Module - src/ocr/__init__.py

# This file makes 'ocr' a Python sub-package.
# It can be used to expose the main functionalities of the OCR module.

# Example imports (actual classes/functions will be defined in other files):
# from .infer import OCRInferencer
# from .train import OCRTrainer
# from .utils import preprocess_image_for_ocr, visualize_ocr_results
# from .postprocess import aggregate_ocr_outputs

VERSION = "0.1.0"

print(f"OCR module (version {VERSION}) initialized.")

# It's good practice to define what gets imported when `from .ocr import *` is used,
# though explicit imports are generally preferred.
# __all__ = ['OCRInferencer', 'OCRTrainer', 'preprocess_image_for_ocr']
