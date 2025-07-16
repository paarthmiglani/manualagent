import json
import os

def coco_to_yolo(coco_json, output_dir, img_dir):
    with open(coco_json) as f:
        coco = json.load(f)

    img_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    img_wh = {img['id']: (img['width'], img['height']) for img in coco['images']}
    os.makedirs(output_dir, exist_ok=True)

    anns_by_img = {}
    for ann in coco['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    for img_id, filename in img_id_to_filename.items():
        w, h = img_wh[img_id]
        label_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
        anns = anns_by_img.get(img_id, [])
        with open(label_path, "w") as f:
            for ann in anns:
                x, y, bw, bh = ann['bbox']
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                bw /= w
                bh /= h
                category = ann['category_id'] - 1  # YOLO expects zero-based class idx
                f.write(f"{category} {xc} {yc} {bw} {bh}\n")
    print(f"Converted COCO to YOLO TXT in {output_dir}")

if __name__ == "__main__":
    coco_json = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox/detection/instances.json"
    output_dir = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox/labels"
    img_dir = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox/images"
    coco_to_yolo(coco_json, output_dir, img_dir)


