import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import time
import os

from distortions.model.custom_resnet import CustomResNet
from distortions.utils.functions import get_backbone_and_weights, train_epoch, validate_epoch
from distortions.dataset.dataset import get_dataloaders  

def main(model, backbone, train_loader, val_loader, device, num_epochs, lr, wandb_enable):
    best_acc = 0.0
    best_loss = float("inf")
    model_path = ""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    wandb.init(
        mode="online" if wandb_enable else "disabled",
        project="distortions-detect",
        config={
            "architecture": type(model.backbone).__name__,
            "epochs": num_epochs,
            "learning_rate": lr,
            "batch_size": train_loader.batch_size,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
            "dataset": "ECSIQ",
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
        },
        name=f"training_{backbone}"
    )

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = f"runs/train/{time_stamp}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/config.yaml", "w") as file:
        for key, value in wandb.config.items():
            file.write(f"{key}: {value}\n")

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss, train_acc, train_precision, train_recall  = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall = validate_epoch(model, val_loader, criterion, device)

        print(f"  ➤ Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Precision: {train_precision:.2f}%, Train Recall: {train_recall:.2f}% "
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Vall Precision: {val_precision:.2f}%, Vall Recall: {val_recall:.2f}%")

        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_acc > best_acc and val_loss < best_loss:
            best_acc, best_loss = val_acc, val_loss
            # Take the batch size from the DataLoader
            model_path = f"{save_dir}/best_distortion_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    wandb.finish()
    return model_path


def train_model(
    backbone='resnet_50',
    data_dir="/home/jmn/host/dev/Datasets/IQA/ECSIQ/",
    train_split=0.7,
    image_shape=(256, 256),
    batch_size=32,
    lr=1e-4,
    num_epochs=10,
    wandb_enable=True
):
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir, 
        train_split=train_split, 
        image_shape=image_shape, 
        batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    back, weights = get_backbone_and_weights(name_model=backbone)

    model = CustomResNet(
        num_classes=7,
        backbone=back,
        weights=weights
    ).to(device)

    return main(model, backbone, train_loader, val_loader, device, num_epochs, lr, wandb_enable)


    