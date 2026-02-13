import torch.nn as nn
from torchvision import models

# 1. SIMPLE CNN (very small)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# 2. COMPLEX CNN (bigger)

class ComplexCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# 3. RESNET18
def resnet18_model(num_classes=2, freeze=True):
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )
    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 4. EFFICIENTNET-B0
def efficientnet_b0_model(num_classes=2, freeze=True):
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    if freeze:
        for p in model.features.parameters():
            p.requires_grad = False

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model

# 5. DENSENET121
def densenet121_model(num_classes=2, freeze=True):
    model = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1
    )
    if freeze:
        for p in model.features.parameters():
            p.requires_grad = False

    model.classifier = nn.Linear(
        model.classifier.in_features, num_classes
    )
    return model