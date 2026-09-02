import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_model(num_classes=3):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
