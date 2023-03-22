import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import transformers
from transformers import (
    ConvNextForImageClassification,
    BeitForImageClassification,
    ViTForImageClassification,
    AutoModelForImageClassification,
)

from timm import create_model


class ConvNext_base(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        model = ConvNextForImageClassification.from_pretrained(
            "facebook/convnext-tiny-224"
        )
        self.backbone = model
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x).logits
        x = F.sigmoid(self.classifier(x))
        return x


class Finetuned_VIT(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        model = ViTForImageClassification.from_pretrained(
            "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
        )
        self.backbone = model

    def forward(self, x):
        x = self.backbone(x).logits
        x = F.sigmoid(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.model(x)
        x = F.sigmoid(x)
        return
