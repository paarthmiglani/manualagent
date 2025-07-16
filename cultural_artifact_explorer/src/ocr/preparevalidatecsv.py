import os
import glob
import csv

IMAGES_DIR = "/Users/paarthmiglani/PycharmProjects/manualagent/validation_images"
LABELS_DIR = "/Users/paarthmiglani/PycharmProjects/manualagent/validation_text"
OUT_CSV = "data/ocr/annotations_val.csv"

rows = []
all_chars = set()

for label_file in glob.glob(os.path.join(LABELS_DIR, "gt_img_*.txt")):
    img_name = os.path.basename(label_file).replace("gt_", "").replace(".txt", ".jpg")
    img_path = os.path.join(IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        continue
    with open(label_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            word = parts[-1]
            if not word.strip():
                continue
            rows.append([img_name, word])
            all_chars.update(list(word))

# Save annotation CSV
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "text"])
    writer.writerows(rows)

# Save character list
with open("data/ocr/char_val_list.txt", "w", encoding="utf-8") as f:
    for c in sorted(all_chars):
        f.write(c + "\n")

print(f"Done! {len(rows)} annotation rows written.")
print(f"Char list has {len(all_chars)} unique characters.")
