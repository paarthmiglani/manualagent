import os
import cv2
import xml.etree.ElementTree as ET

DATA_DIR = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox"
ANNOT_PATH = os.path.join(DATA_DIR, "annotations.xml")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
OUT_DIR = os.path.join(DATA_DIR, "vis")
os.makedirs(OUT_DIR, exist_ok=True)

tree = ET.parse(ANNOT_PATH)
root = tree.getroot()

for image in root.findall("image"):
    img_name = image.attrib["name"]
    img_path = os.path.join(IMAGE_DIR, os.path.basename(img_name))
    img = cv2.imread(img_path)
    if img is None:
        continue
    for box in image.findall("box"):
        label = box.attrib["label"]
        xtl = int(float(box.attrib["xtl"]))
        ytl = int(float(box.attrib["ytl"]))
        xbr = int(float(box.attrib["xbr"]))
        ybr = int(float(box.attrib["ybr"]))
        color = (0, 255, 0)  # green for all
        cv2.rectangle(img, (xtl, ytl), (xbr, ybr), color, 2)
        cv2.putText(img, label, (xtl, ytl - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    vis_path = os.path.join(OUT_DIR, os.path.basename(img_name))
    cv2.imwrite(vis_path, img)
    print(f"Saved visualization: {vis_path}")
