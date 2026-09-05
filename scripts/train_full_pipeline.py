import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import ICBHICycleDataset, ICBHIAudioDataset
from src.models import get_pretrained_cnn
from tqdm import tqdm

def train_model(model, dataloader, criterion, optimizer, device, epochs=3, pipeline_name="Pipeline"):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"[{pipeline_name}] Epoch {epoch+1}/{epochs}")
        for specs, labels in progress_bar:
            specs, labels = specs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        epoch_loss = running_loss / len(dataloader)
        print(f"[{pipeline_name}] Epoch {epoch+1} Completed. Average Loss: {epoch_loss:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training full models on: {device}")
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Train Pipeline B (Cycle-Level)
    print("\n--- Starting Pipeline B (Cycle-Level) Training ---")
    dataset_b = ICBHICycleDataset(index_csv="data/processed/file_index.csv")
    train_size_b = int(0.8 * len(dataset_b))
    val_size_b = len(dataset_b) - train_size_b
    train_set_b, val_set_b = random_split(dataset_b, [train_size_b, val_size_b], generator=torch.Generator().manual_seed(42))
    
    train_loader_b = DataLoader(train_set_b, batch_size=32, shuffle=True, num_workers=2)
    
    model_b = get_pretrained_cnn(model_name='resnet18', num_classes=4, pretrained=True).to(device)
    optimizer_b = optim.Adam(model_b.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_model(model_b, train_loader_b, criterion, optimizer_b, device, epochs=3, pipeline_name="Pipeline B")
    torch.save(model_b.state_dict(), "checkpoints/resnet18_pipeline_b.pth")
    print("Pipeline B checkpoint saved.")
    
    # 2. Train Pipeline A (Whole-Audio)
    print("\n--- Starting Pipeline A (Whole-Audio Baseline) Training ---")
    dataset_a = ICBHIAudioDataset(index_csv="data/processed/file_index.csv", target_duration=10.0)
    train_size_a = int(0.8 * len(dataset_a))
    val_size_a = len(dataset_a) - train_size_a
    train_set_a, val_set_a = random_split(dataset_a, [train_size_a, val_size_a], generator=torch.Generator().manual_seed(42))
    
    train_loader_a = DataLoader(train_set_a, batch_size=8, shuffle=True, num_workers=2)
    
    model_a = get_pretrained_cnn(model_name='resnet18', num_classes=4, pretrained=True).to(device)
    optimizer_a = optim.Adam(model_a.parameters(), lr=1e-4)
    
    train_model(model_a, train_loader_a, criterion, optimizer_a, device, epochs=3, pipeline_name="Pipeline A")
    torch.save(model_a.state_dict(), "checkpoints/resnet18_pipeline_a.pth")
    print("Pipeline A checkpoint saved.")
    
    print("\nAll training complete. Checkpoints saved to checkpoints/")

if __name__ == '__main__':
    main()