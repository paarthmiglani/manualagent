import torch
from src.ocr.model_definition import CRNN
from src.ocr.dataset import OCRDataset
from src.ocr.utils import load_char_list, preprocess_image_for_ocr
from src.ocr.postprocess import ctc_decode_predictions
import matplotlib.pyplot as plt
import argparse
import os

def show_prediction(image_path, pred_text, true_text):
    import cv2
    img = cv2.imread(image_path)
    plt.figure(figsize=(8, 4))
    plt.imshow(img[:,:,::-1])
    plt.title(f"GT: {true_text}\nPred: {pred_text}")
    plt.axis("off")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ocr.yaml")
    parser.add_argument("--model", default="models/ocr/crnn_epoch_1.pth")
    parser.add_argument("--csv", default="YOUR_VAL_CSV_PATH")  # change this or pass as arg
    parser.add_argument("--img_dir", default="YOUR_IMG_DIR")   # change this or pass as arg
    parser.add_argument("--char_list", default="PATH_TO_CHAR_LIST")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    # Load char list
    char_list = load_char_list(args.char_list)
    ix_to_char = {i + 1: char for i, char in enumerate(char_list)}
    ix_to_char[0] = "<BLANK>"

    # Load dataset
    dataset = OCRDataset(
        annotations_file=args.csv,
        img_dir=args.img_dir,
        char_list_path=args.char_list,
        image_height=32,
        image_width=256,
    )

    # Load model
    num_classes = len(char_list) + 1
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CRNN(img_channels=1, num_classes=num_classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    model.to(device)

    # Show a few predictions
    for i in range(args.num_samples):
        image_tensor, label_tensor = dataset[i]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model(image_tensor)
            pred = log_probs.permute(1, 0, 2).cpu().numpy()[0] # (T, C)
            pred_text, _ = ctc_decode_predictions(pred, char_list, blank_idx=0)
        # True text
        label_str = "".join([ix_to_char[ix.item()] for ix in label_tensor])
        # Show image with prediction and GT
        image_path = dataset.annotations.iloc[i]['filepath'] if 'filepath' in dataset.annotations.columns else os.path.join(args.img_dir, dataset.annotations.iloc[i]['filename'])
        show_prediction(image_path, pred_text, label_str)

if __name__ == "__main__":
    main()

