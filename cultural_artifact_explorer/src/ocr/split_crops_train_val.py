import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

CROP_DIR = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox/crops"
LABEL_CSV = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox/recog_labels.csv"
OUT_DIR_TRAIN = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox/train_crops"
OUT_DIR_VAL = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox/val_crops"
TRAIN_CSV = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox/train_recog_labels.csv"
VAL_CSV = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/boundingbox/val_recog_labels.csv"

os.makedirs(OUT_DIR_TRAIN, exist_ok=True)
os.makedirs(OUT_DIR_VAL, exist_ok=True)

df = pd.read_csv(LABEL_CSV)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
for subset, subset_df, out_dir in [("train", train_df, OUT_DIR_TRAIN), ("val", val_df, OUT_DIR_VAL)]:
    for _, row in subset_df.iterrows():
        src = os.path.join(CROP_DIR, row['filename'])
        dst = os.path.join(out_dir, row['filename'])
        if os.path.exists(src):
            shutil.copy(src, dst)
subset_dfs = {"train": train_df, "val": val_df}
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
print(f"Train: {len(train_df)} | Val: {len(val_df)} samples")
