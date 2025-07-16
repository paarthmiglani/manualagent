import os
from PIL import Image

LABELS_DIR = '/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/images/labels'
IMAGES_DIR = '/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/images'
YOLO_LABELS_DIR = os.path.join(LABELS_DIR, "labels")
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

for filename in os.listdir(LABELS_DIR):
    if not filename.endswith('.txt') or filename.startswith('labels'):
        continue

    txt_path = os.path.join(LABELS_DIR, filename)
    img_name = filename.replace('.txt', '.jpg')
    img_path = os.path.join(IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        img_name = filename.replace('.txt', '.png')
        img_path = os.path.join(IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"Image not found for {filename}, skipping.")
        continue

    img = Image.open(img_path)
    W, H = img.size

    yolo_lines = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 10:
                continue
            try:
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                # Get min/max box
                x_min = min(x1, x2, x3, x4)
                x_max = max(x1, x2, x3, x4)
                y_min = min(y1, y2, y3, y4)
                y_max = max(y1, y2, y3, y4)
                # Convert to YOLO format
                cx = ((x_min + x_max) / 2) / W
                cy = ((y_min + y_max) / 2) / H
                bw = (x_max - x_min) / W
                bh = (y_max - y_min) / H
                # Always class 0 for "text"
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            except Exception as e:
                print(f"Error processing line in {filename}: {e}")
                continue

    out_txt = os.path.join(YOLO_LABELS_DIR, filename.replace('.txt', '.txt'))
    with open(out_txt, 'w', encoding="utf-8") as f:
        f.write('\n'.join(yolo_lines))
    print(f"Converted {filename} to YOLO format.")

print("All GT files converted to YOLO format in:", YOLO_LABELS_DIR)
