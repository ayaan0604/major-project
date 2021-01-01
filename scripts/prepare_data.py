import os
import zipfile
import pandas as pd
import tqdm

# 1. Extract raw zip file if not already extracted
raw_data_dir = "data/raw_data"
extracted_dir = os.path.join(raw_data_dir, "extracted")
os.makedirs(extracted_dir, exist_ok=True)

for file in tqdm.tqdm(os.listdir(raw_data_dir)):
    if file.endswith(".zip"):
        zip_path = os.path.join(raw_data_dir, file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        print(f"Extracted: {file}")

# 2. Locate and index all valid audio-annotation pairs
def locate_icbhi_files(base_dir):
    wav_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
                
    records = []
    for wav_path in wav_files:
        txt_path = os.path.splitext(wav_path)[0] + '.txt'
        if os.path.exists(txt_path):
            records.append({
                'wav_path': wav_path,
                'txt_path': txt_path,
                'filename': os.path.basename(wav_path)
            })
            
    return pd.DataFrame(records)

os.makedirs("data/processed", exist_ok=True)
dataset_index = locate_icbhi_files(extracted_dir)
print(f"Successfully indexed {len(dataset_index)} valid audio-annotation pairs.")
dataset_index.to_csv("data/processed/file_index.csv", index=False)