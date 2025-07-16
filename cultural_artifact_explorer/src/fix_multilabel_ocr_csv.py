import csv

# Path to your original, broken CSV
INPUT_CSV = '/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/annotations.csv'
# Path to the new, fixed CSV
OUTPUT_CSV = '/Users/paarthmiglani/PycharmProjects/manualagent/data/ocr/annotations_fixed.csv'

with open(INPUT_CSV, newline='', encoding='utf-8') as infile, \
        open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    # Define the new header. Adjust if you have different box formats (here 8-point poly).
    fieldnames = [
        'filename', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'script', 'text'
    ]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        filename = row['filename']
        text_blob = row['text']
        # Split by line, each line is a region label for that image
        region_lines = [l for l in text_blob.split('\n') if l.strip()]
        for region in region_lines:
            parts = region.strip().split(',')
            if len(parts) < 11:
                # skip malformed lines
                continue
            region_data = {
                'filename': filename,
                'x1': parts[0],
                'y1': parts[1],
                'x2': parts[2],
                'y2': parts[3],
                'x3': parts[4],
                'y3': parts[5],
                'x4': parts[6],
                'y4': parts[7],
                'script': parts[8],
                'text': ','.join(parts[9:]).strip(),  # handles commas in label text
            }
            writer.writerow(region_data)

print(f"Fixed CSV written to: {OUTPUT_CSV}")
