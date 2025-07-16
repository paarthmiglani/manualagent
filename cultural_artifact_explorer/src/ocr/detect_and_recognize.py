import os
import argparse
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from ultralytics import YOLO
import numpy as np
from src.ocr.model_definition import CRNN

# --- Utility: Load char list ---
def load_char_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        chars = [line.strip('\r\n') for line in f.readlines() if line.strip()]
    return chars

# --- Preprocess crop for CRNN ---
def preprocess_crop(pil_img, img_height=32):
    # Grayscale, resize keeping aspect ratio
    pil_img = pil_img.convert('L')
    w, h = pil_img.size
    new_w = max(8, int(w * img_height / h))
    pil_img = pil_img.resize((new_w, img_height), Image.BILINEAR)
    transform = T.Compose([
        T.ToTensor(),                   # [1, H, W], 0-1
        T.Normalize([0.5], [0.5])       # [-1, 1]
    ])
    tensor = transform(pil_img).unsqueeze(0)  # [1, 1, H, W]
    return tensor

# --- Simple greedy CTC decoder ---
def ctc_greedy_decode(log_probs, char_list):
    preds = log_probs.argmax(2)  # [seq, batch]
    texts = []
    for b in range(preds.shape[1]):
        pred = preds[:, b].cpu().numpy()
        prev = -1
        out = []
        for idx in pred:
            if idx != prev and idx < len(char_list):
                out.append(char_list[idx])
            prev = idx
        texts.append(''.join(out))
    return texts

# --- Draw boxes and texts ---
def draw_boxes(image, boxes, texts):
    draw = ImageDraw.Draw(image)
    font = None
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    for box, text in zip(boxes, texts):
        box = [tuple(box[:2]), tuple(box[2:])]
        draw.rectangle(box, outline='red', width=2)
        draw.text(box[0], text, fill='blue', font=font)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', type=str, required=False,
                        default='runs/detect/train7/weights/best_1.pt')
    parser.add_argument('--crnn_model', type=str, required=False,
                        default='models/ocr/crnn_epoch_100.pth')
    parser.add_argument('--char_list_path', type=str, required=True)
    parser.add_argument('--image', type=str, required=True, help="Image file or directory")
    parser.add_argument('--save_dir', type=str, default='ocr_results')
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--input_height', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load char list
    char_list = load_char_list(args.char_list_path)
    num_classes = len(char_list) + 1  # For CTC blank

    # Load YOLO detector
    yolo = YOLO(args.yolo_model)
    print(f"Loaded YOLO model from {args.yolo_model}")

    # Load CRNN recognizer
    crnn = CRNN(img_channels=1, num_classes=num_classes)
    crnn.load_state_dict(torch.load(args.crnn_model, map_location=args.device))
    crnn.eval()
    crnn.to(args.device)
    print(f"Loaded CRNN model from {args.crnn_model}")

    # Handle single image or directory
    if os.path.isdir(args.image):
        image_files = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.lower().endswith(('jpg','jpeg','png','bmp'))]
    else:
        image_files = [args.image]

    for img_path in image_files:
        print(f"\nProcessing {img_path}")
        im = Image.open(img_path).convert('RGB')
        w, h = im.size

        # --- Detection ---
        results = yolo(img_path)[0]
        if len(results.boxes) == 0:
            print("No text boxes detected.")
            continue
        boxes_xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
        # Each box: [xmin, ymin, xmax, ymax]

        recognized = []
        for box in boxes_xyxy:
            # Clamp coordinates
            x1,y1,x2,y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = im.crop((x1, y1, x2, y2))
            tensor = preprocess_crop(crop, img_height=args.input_height).to(args.device)
            with torch.no_grad():
                log_probs = crnn(tensor)
            text = ctc_greedy_decode(log_probs, char_list)[0]
            recognized.append(text)
            print(f"Box: [{x1}, {y1}, {x2}, {y2}] Text: '{text}'")

        # Draw boxes and save result
        out_img = im.copy()
        boxes_for_draw = [ [box[0], box[1], box[2], box[3]] for box in boxes_xyxy ]
        out_img = draw_boxes(out_img, boxes_for_draw, recognized)
        base = os.path.basename(img_path)
        out_path = os.path.join(args.save_dir, f"ocr_{base}")
        out_img.save(out_path)
        print(f"OCR result saved to {out_path}")

if __name__ == "__main__":
    main()
