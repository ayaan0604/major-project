import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import ICBHIAudioDataset
from src.models import get_pretrained_cnn
from tqdm import tqdm

def train_pipeline_a():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Pipeline A (Whole-Audio Baseline) on: {device}")
    
    # Initialize Whole-Audio Dataset
    dataset = ICBHIAudioDataset(index_csv="data/processed/file_index.csv", target_duration=10.0)
    
    # Train/Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # Use smaller batch size (8) since whole-audio spectrograms have larger time dimensions
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2)
    
    # Model, Loss, Optimizer
    model = get_pretrained_cnn(model_name='resnet18', num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 1-Epoch Sanity Check Training Loop for Pipeline A
    model.train()
    running_loss = 0.0
    for specs, labels in tqdm(train_loader, desc="Training Pipeline A"):
        specs, labels = specs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Pipeline A Epoch Complete. Average Training Loss: {running_loss / len(train_loader):.4f}")

if __name__ == '__main__':
    train_pipeline_a()