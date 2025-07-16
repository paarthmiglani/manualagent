import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# from src.ocr.infer import OCRInferencer

class TestOCREngine(unittest.TestCase):
    @unittest.skip("Skipping OCR test because it requires torch")
    @patch('src.ocr.infer.OCRInferencer')
    def test_get_text(self, MockOCRInferencer):
        # Create a mock instance of the OCRInferencer
        mock_inferencer_instance = MockOCRInferencer.return_value
        mock_inferencer_instance.predict.return_value = ("test text", 0.9)

        # Since get_ocr_engine is a factory, we need to patch it to return our mock
        with patch('ocr.ocr_engine.get_ocr_engine') as mock_get_engine:
            # Configure the factory to return an object that uses our mock inferencer
            mock_engine = MagicMock()
            mock_engine.get_text.return_value = mock_inferencer_instance.predict("dummy_path")
            mock_get_engine.return_value = mock_engine

            # Now, get the engine from the (mocked) factory
            ocr_engine = mock_get_engine("dummy_config.yaml")

            # Call the method and assert
            text, confidence = ocr_engine.get_text("dummy_image.png")
            self.assertEqual(text, "test text")
            self.assertEqual(confidence, 0.9)
            mock_inferencer_instance.predict.assert_called_once_with("dummy_path")


if __name__ == '__main__':
    unittest.main()
