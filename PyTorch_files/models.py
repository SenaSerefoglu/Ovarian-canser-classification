import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Custom CNN Model
class CNNmodel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNmodel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.4)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 14 * 14, 64)  # Assuming input size is 224x224
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


# ResNet50 Custom Model
def ResNet50CustomModel(num_classes=5):
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Sequential(
        nn.Linear(base_model.fc.in_features, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.4),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.4),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.4),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.BatchNorm1d(16),
        nn.Dropout(0.4),
        nn.Linear(16, num_classes),
    )
    return base_model


# VGG19 Custom Model
def VGG19CustomModel(num_classes=5):
    base_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    base_model.classifier = nn.Sequential(
        nn.Linear(base_model.classifier[0].in_features, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.4),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.4),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.4),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.BatchNorm1d(16),
        nn.Dropout(0.4),
        nn.Linear(16, num_classes),
    )
    return base_model


# VGG16 Custom Model
def VGG16CustomModel(num_classes=5):
    base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    base_model.classifier = nn.Sequential(
        nn.Linear(base_model.classifier[0].in_features, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.4),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.4),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.4),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.BatchNorm1d(16),
        nn.Dropout(0.4),
        nn.Linear(16, num_classes),
    )
    return base_model
