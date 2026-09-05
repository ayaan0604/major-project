import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.dataset import ICBHIAudioDataset, ICBHICycleDataset
from src.models import get_pretrained_cnn
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import os
import pandas as pd
import numpy as np

def generate_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating comprehensive evaluation metrics on: {device}")
    
    # Target Class Names
    class_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    
    # 1. Pipeline A Metrics
    dataset_a = ICBHIAudioDataset(index_csv="data/processed/file_index.csv", target_duration=10.0)
    train_size_a = int(0.8 * len(dataset_a))
    val_size_a = len(dataset_a) - train_size_a
    _, val_set_a = random_split(dataset_a, [train_size_a, val_size_a], generator=torch.Generator().manual_seed(42))
    val_loader_a = DataLoader(val_set_a, batch_size=8, shuffle=False, num_workers=2)
    
    model_a = get_pretrained_cnn(model_name='resnet18', num_classes=4, pretrained=False).to(device)
    model_a.load_state_dict(torch.load("checkpoints/resnet18_pipeline_a.pth", map_location=device))
    model_a.eval()
    
    y_true_a, y_pred_a = [], []
    with torch.no_grad():
        for specs, labels in val_loader_a:
            specs = specs.to(device)
            outputs = model_a(specs)
            _, predicted = torch.max(outputs.data, 1)
            y_true_a.extend(labels.numpy())
            y_pred_a.extend(predicted.cpu().numpy())
            
    print("\n--- Pipeline A (Baseline Whole-Audio) Classification Report ---")
    print(classification_report(y_true_a, y_pred_a, target_names=class_names, zero_division=0))

    # 2. Pipeline B Metrics (Aggregated File-Level)
    dataset_b = ICBHICycleDataset(index_csv="data/processed/file_index.csv")
    model_b = get_pretrained_cnn(model_name='resnet18', num_classes=4, pretrained=False).to(device)
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
        for idx in range(len(dataset_b)):
            spec, _ = dataset_b[idx]
            sample_info = dataset_b.samples[idx]
            wav_path = sample_info['wav_path']
            
            spec = spec.unsqueeze(0).to(device)
            output = model_b(spec)
            pred_class = torch.argmax(output, dim=1).item()
            
            if wav_path not in file_predictions:
                file_predictions[wav_path] = []
            file_predictions[wav_path].append(pred_class)
            
    y_true_b, y_pred_b = [], []
    for wav_path, preds in file_predictions.items():
        if wav_path not in file_ground_truth:
            continue
        majority_pred = Counter(preds).most_common(1)[0][0]
        y_true_b.append(file_ground_truth[wav_path])
        y_pred_b.append(majority_pred)
        
    print("\n--- Pipeline B (Proposed Cycle-Level + Majority Vote) Classification Report ---")
    print(classification_report(y_true_b, y_pred_b, target_names=class_names, zero_division=0))

if __name__ == '__main__':
    generate_metrics()