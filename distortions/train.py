import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from distortions.model.custom_resnet import CustomResNet
from distortions.utils.functions import train_epoch, validate_epoch
from distortions.dataset.dataset import get_dataloaders  # supondo que você tenha essa função
from torchvision import models

def main(model, backbone, train_loader, val_loader, device, num_epochs, lr):
    best_acc = 0.0
    best_loss = float("inf")
    model_path = ""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Inicializa o W&B ---
    wandb.init(
        project="distortions-detect",
        config={
            "architecture": type(model).__name__,
            "epochs": num_epochs,
            "learning_rate": lr,
            "batch_size": train_loader.batch_size,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
            "dataset": "ECSIQ",
        },
        name=f"training_{backbone}"
    )

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # --- Log no W&B ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # --- Salva o melhor modelo ---
        if val_acc > best_acc and val_loss < best_loss:
            best_acc = val_acc
            best_loss = val_loss
            model_path = f"best_distortions_{epoch+1}_{backbone}_b16_lr1e-4.pth"
            torch.save(model.state_dict(), model_path)

    wandb.finish()
    return model_path


def train_model(
    backbone='resnet_50',
    data_dir="/home/jmn/host/dev/Datasets/IQA/ECSIQ/",
    train_split=0.7,
    image_shape=(256, 256),
    batch_size=16,
    lr=1e-4,
    num_epochs=10
):
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir, 
        train_split=train_split, 
        image_shape=image_shape, 
        batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if backbone == 'resnet_50':
        back = models.resnet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    elif backbone == 'resnet_101':
        back = models.resnet101
        weights = models.ResNet101_Weights.IMAGENET1K_V2
    elif backbone == 'resnet_152':
        back = models.resnet152
        weights = models.ResNet152_Weights.IMAGENET1K_V2
    else:
        raise ValueError(f"Modelo desconhecido: {model}, escolha entre 'resnet_50', 'resnet_101' ou 'resnet_152'.")

    model = CustomResNet(
        num_classes=7,
        backbone=back,
        weights=weights
    ).to(device)

    return main(model, backbone, train_loader, val_loader, device, num_epochs, lr)


    