import pandas as pd
import os

csv_path = "/Users/paarthmiglani/PycharmProjects/manualagent/cultural_artifact_explorer/scripts/data/ocr/annotations.csv"
img_dir = "/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/images"

df = pd.read_csv(csv_path)
header = df.columns[0]
missing = []

for img_name in df[header]:
    img_path = os.path.join(img_dir, str(img_name))
    if not os.path.exists(img_path):
        missing.append(img_name)

print(f"Total images in CSV: {len(df)}")
print(f"Missing images: {len(missing)}")
if missing:
    print("Some missing image filenames (first 10):", missing[:10])
else:
    print("All images referenced in CSV exist in the directory!")



