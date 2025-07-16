import os
import xml.etree.ElementTree as ET
import json

DATA_DIR = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox"
ANNOT_PATH = os.path.join(DATA_DIR, "annotations.xml")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_JSON = os.path.join(DATA_DIR, "detection", "instances.json")
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

categories = [{"id": 1, "name": "text"}]
images = []
annotations = []

tree = ET.parse(ANNOT_PATH)
root = tree.getroot()
ann_id = 1

for img_id, image in enumerate(root.findall("image")):
    img_name = image.attrib["name"]
    img_path = os.path.join(IMAGE_DIR, os.path.basename(img_name))
    if not os.path.exists(img_path):
        continue
    width = int(image.attrib["width"])
    height = int(image.attrib["height"])
    images.append({
        "id": img_id,
        "file_name": os.path.basename(img_name),
        "width": width,
        "height": height
    })
    for box in image.findall("box"):
        xtl = float(box.attrib["xtl"])
        ytl = float(box.attrib["ytl"])
        xbr = float(box.attrib["xbr"])
        ybr = float(box.attrib["ybr"])
        w = xbr - xtl
        h = ybr - ytl
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [xtl, ytl, w, h],
            "area": w * h,
            "iscrowd": 0
        })
        ann_id += 1

coco = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco, f)
print(f"COCO detection dataset saved to {OUTPUT_JSON}")
