import os
import cv2
import xml.etree.ElementTree as ET
import pandas as pd

DATA_DIR = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox"
ANNOT_PATH = os.path.join(DATA_DIR, "annotations.xml")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CROP_DIR = os.path.join(DATA_DIR, "crops")
os.makedirs(CROP_DIR, exist_ok=True)

records = []

tree = ET.parse(ANNOT_PATH)
root = tree.getroot()

for image in root.findall("image"):
    img_name = image.attrib["name"]
    img_path = os.path.join(IMAGE_DIR, os.path.basename(img_name))
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}, skipping")
        continue
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        continue
    for box in image.findall("box"):
        label = box.attrib["label"]
        xtl = float(box.attrib["xtl"])
        ytl = float(box.attrib["ytl"])
        xbr = float(box.attrib["xbr"])
        ybr = float(box.attrib["ybr"])
        text = ""
        for attr in box.findall("attribute"):
            if attr.attrib.get("name") == "text":
                text = attr.text or ""
        crop = img[int(ytl):int(ybr), int(xtl):int(xbr)]
        crop_filename = f"{os.path.splitext(os.path.basename(img_name))[0]}_{label}_{int(xtl)}_{int(ytl)}.jpg"
        crop_path = os.path.join(CROP_DIR, crop_filename)
        cv2.imwrite(crop_path, crop)
        records.append({"filename": crop_filename, "text": text})

df = pd.DataFrame(records)
recog_labels_path = os.path.join(DATA_DIR, "recog_labels.csv")
df.to_csv(recog_labels_path, index=False)
print(f"Saved {len(df)} crops and recog_labels.csv to {CROP_DIR}")
