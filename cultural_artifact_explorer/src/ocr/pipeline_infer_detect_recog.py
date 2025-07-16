import torch
import cv2
import numpy as np
from ultralytics import YOLO
from src.ocr.model_definition import CRNN
from src.ocr.utils import load_char_list, preprocess_image_for_ocr
import sys

# CONFIGURABLE PATHS (edit as needed)
YOLO_MODEL = "runs/detect/train/weights/best.pt"
OCR_MODEL = "models/ocr_bbox/model_best.pth"
CHAR_LIST_PATH = "/Users/paarthmiglani/PycharmProjects/manualagent/cultural_artifact_explorer/scripts/data/ocr/char_list.txt"

def get_img_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return input("Enter path to image for end-to-end OCR: ").strip()

def ctc_decode(preds, ix_to_char):
    pred_indices = torch.argmax(preds, dim=2).squeeze(1).cpu().numpy()
    last_idx = -1
    decoded = ""
    for idx in pred_indices:
        if idx != last_idx and idx != 0:
            decoded += ix_to_char.get(idx, "")
        last_idx = idx
    return decoded

def main():
    IMG_PATH = get_img_path()
    assert IMG_PATH and os.path.exists(IMG_PATH), f"Image not found: {IMG_PATH}"

    # 1. Detect text regions
    yolo = YOLO(YOLO_MODEL)
    results = yolo(IMG_PATH)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    image = cv2.imread(IMG_PATH)

    # 2. OCR model
    char_list = load_char_list(CHAR_LIST_PATH)
    ix_to_char = {i+1: c for i, c in enumerate(char_list)}
    ix_to_char[0] = '<BLANK>'
    num_classes = len(char_list) + 1
    ocr_model = CRNN(1, num_classes, 256, 2, 0.5)
    ocr_model.load_state_dict(torch.load(OCR_MODEL, map_location='cpu'))
    ocr_model.eval()

    # 3. For each detected box, crop and recognize
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        temp_file = "temp_crop.jpg"
        cv2.imwrite(temp_file, crop)
        img = preprocess_image_for_ocr(temp_file, target_size=(256, 32), binarize=False)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        with torch.no_grad():
            preds = ocr_model(img_tensor)
        text = ctc_decode(preds.permute(1, 0, 2), ix_to_char)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        print(f"Detected: {text}")

    out_file = "demo_with_ocr.jpg"
    cv2.imwrite(out_file, image)
    print(f"Saved end-to-end result as {out_file}")

if __name__ == "__main__":
    import os
    main()
