import pandas as pd
import random
from tqdm import tqdm
from src.ocr.utils import load_char_list
from src.ocr.model_definition import CRNN
import torch
from src.ocr.dataset import preprocess_image_for_ocr

# Paths
CROP_DIR = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox/crops"
LABEL_CSV = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox/recog_labels.csv"
CHAR_LIST_PATH = "/Users/paarthmiglani/PycharmProjects/manualagent/cultural_artifact_explorer/scripts/data/ocr/char_list.txt"
MODEL_PATH = "models/ocr_bbox/crnn_epoch_100.pth"   # Adjust as needed

def ctc_decode(preds, ix_to_char):
    # preds: (seq_len, batch, num_classes)
    pred_indices = torch.argmax(preds, dim=2).cpu().numpy()  # (seq_len, batch)
    decoded_batch = []
    # Handle each item in batch
    for batch_idx in range(pred_indices.shape[1]):
        last_idx = -1
        decoded = ""
        for idx in pred_indices[:, batch_idx]:
            if idx != last_idx and idx != 0:
                decoded += ix_to_char.get(idx, "")
            last_idx = idx
        decoded_batch.append(decoded)
    return decoded_batch[0]  # if batch size = 1, else return list


def cer(s1, s2):
    # Simple character error rate
    from Levenshtein import distance as levenshtein
    return levenshtein(s1, s2) / max(1, len(s1))

def wer(s1, s2):
    # Simple word error rate
    from jiwer import wer as jiwer_wer
    return jiwer_wer(s1, s2)

def main(num_samples=20):
    df = pd.read_csv(LABEL_CSV)
    char_list = load_char_list(CHAR_LIST_PATH)
    ix_to_char = {i+1: c for i, c in enumerate(char_list)}
    ix_to_char[0] = '<BLANK>'
    num_classes = len(char_list) + 1

    model = CRNN(
        img_channels=1,
        num_classes=num_classes,
        rnn_hidden_size=256,
        rnn_num_layers=2,
        dropout=0.5
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    total_cer, total_wer = 0, 0
    all_samples = df.sample(min(len(df), num_samples), random_state=42).to_dict('records')
    print("Random prediction pairs (Ground Truth vs Pred):\n")
    for sample in tqdm(all_samples):
        img_path = f"{CROP_DIR}/{sample['filename']}"
        gt = str(sample['text'])
        img = preprocess_image_for_ocr(img_path, target_size=(256, 32), binarize=False)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        with torch.no_grad():
            preds = model(img_tensor)
        decoded = ctc_decode(preds.permute(1, 0, 2), ix_to_char)
        sample_cer = cer(gt, decoded)
        sample_wer = wer(gt, decoded)
        total_cer += sample_cer
        total_wer += sample_wer
        print(f"GT: '{gt}' | Pred: '{decoded}' | CER: {sample_cer:.2f} | WER: {sample_wer:.2f}")

    avg_cer = total_cer / len(all_samples)
    avg_wer = total_wer / len(all_samples)
    print(f"\nBatch Average CER: {avg_cer:.3f}, Average WER: {avg_wer:.3f}")

if __name__ == "__main__":
    main(num_samples=20)  # Adjust sample size as desired
