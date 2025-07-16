from ultralytics import YOLO

model_path = "/Users/paarthmiglani/PycharmProjects/manualagent/cultural_artifact_explorer/runs/detect/train7/weights/best_1.pt"
img_dir = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/ch8_validation_images"

model = YOLO(model_path)
results = model(img_dir, save=True, imgsz=640)  # save results to disk