import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights

def get_pretrained_cnn(model_name='resnet18', num_classes=4, pretrained=True):
    """
    Initializes a pretrained vision backbone (ResNet18 or MobileNetV2) 
    modified for 3-channel Mel-spectrogram classification.
    """
    if model_name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
        
    return model