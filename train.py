import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import argparse

import cv2
import pandas as pd
import numpy as np
import os
from copy import deepcopy
import random
import glob
import random

from torchvision import models
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)


from tqdm.auto import tqdm
import wandb
from .util import *

import warnings

warnings.filterwarnings(action="ignore")


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_acc = []
    recall = 0
    preds_list = []
    label_list = []
    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            probs = model(imgs)
            loss = criterion(probs, labels)
            probs = probs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            preds = probs > 0.5
            batch_acc = (labels == preds).mean()

            val_acc.append(batch_acc)
            val_loss.append(loss.item())
            preds_list.extend(preds)
            label_list.extend(labels)

        recall = recall_score(preds_list, label_list, average="weighted")
        f1 = f1_score(preds_list, label_list, average="weighted")
        _val_loss = np.mean(val_loss)
        _val_acc = np.mean(val_acc)

    return _val_loss, _val_acc, recall, f1


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(focal_loss)


def train(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    early_stop = 0

    tr_path = glob.glob("./copy_images/trainset/*.jpg")
    random.shuffle(tr_path)
    tr_labels = [[1] if "ab" in i else [0] for i in tr_path]
    train_transform = A.Compose(
        [
            A.Resize(args.img_size, args.img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=False,
                p=1.0,
            ),
            A.HorizontalFlip(p=1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            A.RandomResizedCrop(
                height=args.img_size, width=args.img_size, scale=(0.3, 1.0)
            ),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Resize(args.img_size, args.img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=False,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )

    te_path = glob.glob("./copy_images/testset/*.jpg")
    random.shuffle(te_path)
    test_labels = [[1] if "ab" in i else [0] for i in te_path]

    train_dataset = CustomDataset(tr_path, tr_labels, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dataset = CustomDataset(te_path, test_labels, test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ResNet()
    model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    best_val_acc = 0
    best_model = None

    criterion = FocalLoss().to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_acc, recall, f1 = validation(
            model, criterion, test_loader, device
        )
        _train_loss = np.mean(train_loss)

        print(
            f"Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_acc:.5f}]"
        )
        log_dic = {}
        log_dic["val loss"] = _val_loss
        log_dic["val acc"] = _val_acc
        log_dic["recall"] = recall
        log_dic["f1 score"] = f1
        wandb.log(log_dic)
        if best_val_acc < _val_acc:
            best_val_acc = _val_acc
            best_model = deepcopy(model)
            early_stop = 0
        else:
            early_stop += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_name", default="Resnet")
    parser.add_argument("--detail", default="cut_mix")
    parser.add_argument("--makecsvfile", type=bool, default=False)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--clip", default=1)
    args = parser.parse_args()

    seed_everything(args.seed)

    wandb.init(
        entity="mini3",
        project=args.model_name,
        name=args.detail,
        config={"epochs": args.epochs, "batch_size": args.batch_size},
    )

    train(args)
