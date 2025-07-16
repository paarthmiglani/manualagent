import pandas as pd

train_csv = "/Users/paarthmiglani/PycharmProjects/manualagent/cultural_artifact_explorer/src/ocr/data/ocr/annotations.csv"
val_csv = "/Users/paarthmiglani/PycharmProjects/manualagent/cultural_artifact_explorer/src/ocr/data/ocr/annotations_val.csv"
output_charlist = "/Users/paarthmiglani/PycharmProjects/manualagent/cultural_artifact_explorer/src/ocr/data/ocr/char_list.txt"

all_chars = set()

for fname in [train_csv, val_csv]:
    df = pd.read_csv(fname)
    for text in df['text']:
        all_chars.update(str(text))

sorted_chars = sorted(all_chars)
with open(output_charlist, "w", encoding="utf-8") as f:
    for c in sorted_chars:
        f.write(c + "\n")

print(f"Wrote {len(sorted_chars)} unique characters to {output_charlist}")
