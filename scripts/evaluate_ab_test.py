import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.dataset import ICBHIAudioDataset, ICBHICycleDataset
from src.models import get_pretrained_cnn
from collections import Counter
import os
import pandas as pd
from tqdm import tqdm

def evaluate_pipelines():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running A/B Evaluation Framework on: {device}")
    
    # 1. Evaluate Pipeline A
    dataset_a = ICBHIAudioDataset(index_csv="data/processed/file_index.csv", target_duration=10.0)
    train_size_a = int(0.8 * len(dataset_a))
    val_size_a = len(dataset_a) - train_size_a
    _, val_set_a = random_split(dataset_a, [train_size_a, val_size_a], generator=torch.Generator().manual_seed(42))
    val_loader_a = DataLoader(val_set_a, batch_size=8, shuffle=False, num_workers=2)
    
    model_a = get_pretrained_cnn(model_name='resnet18', num_classes=4, pretrained=False).to(device)
    if os.path.exists("checkpoints/resnet18_pipeline_a.pth"):
        model_a.load_state_dict(torch.load("checkpoints/resnet18_pipeline_a.pth", map_location=device))
    model_a.eval()
    
    correct_a, total_a = 0, 0
    with torch.no_grad():
        for specs, labels in tqdm(val_loader_a, desc="Evaluating Pipeline A"):
            specs, labels = specs.to(device), labels.to(device)
            outputs = model_a(specs)
            _, predicted = torch.max(outputs.data, 1)
            total_a += labels.size(0)
            correct_a += (predicted == labels).sum().item()
            
    acc_a = 100 * correct_a / total_a if total_a > 0 else 0.0
    print(f"\nPipeline A (Baseline Whole-Audio) Validation Accuracy: {acc_a:.2f}%\n")

    # 2. Evaluate Pipeline B with Majority Vote
    dataset_b = ICBHICycleDataset(index_csv="data/processed/file_index.csv")
    model_b = get_pretrained_cnn(model_name='resnet18', num_classes=4, pretrained=False).to(device)
    if os.path.exists("checkpoints/resnet18_pipeline_b.pth"):
        model_b.load_state_dict(torch.load("checkpoints/resnet18_pipeline_b.pth", map_location=device))
    model_b.eval()
    
    file_ground_truth = {}
    index_df = pd.read_csv("data/processed/file_index.csv")
    for _, row in index_df.iterrows():
        ann_df = pd.read_csv(row['txt_path'], sep='\t', names=['start', 'end', 'crackles', 'wheezes'], header=None)
        c, w = ann_df['crackles'].max(), ann_df['wheezes'].max()
        if c == 0 and w == 0: lbl = 0
        elif c == 1 and w == 0: lbl = 1
        elif c == 0 and w == 1: lbl = 2
        else: lbl = 3
        file_ground_truth[row['wav_path']] = lbl

    file_predictions = {}
    with torch.no_grad():
        for idx in tqdm(range(len(dataset_b)), desc="Evaluating Pipeline B Cycles"):
            spec, _ = dataset_b[idx]
            sample_info = dataset_b.samples[idx]
            wav_path = sample_info['wav_path']
            
            spec = spec.unsqueeze(0).to(device)
            output = model_b(spec)
            pred_class = torch.argmax(output, dim=1).item()
            
            if wav_path not in file_predictions:
                file_predictions[wav_path] = []
            file_predictions[wav_path].append(pred_class)
            
    correct_b, total_b = 0, 0
    for wav_path, preds in file_predictions.items():
        if wav_path not in file_ground_truth:
            continue
        majority_pred = Counter(preds).most_common(1)[0][0]
        true_label = file_ground_truth[wav_path]
        
        if majority_pred == true_label:
            correct_b += 1
        total_b += 1
        
    acc_b = 100 * correct_b / total_b if total_b > 0 else 0.0
    print(f"\nPipeline B (Proposed Cycle-Level + Majority Vote) Validated Accuracy: {acc_b:.2f}%\n")

if __name__ == '__main__':
    evaluate_pipelines()