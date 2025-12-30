import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from distortions.model.noise_model import NoiseClassificationNet
from distortions.model.custom_resnet import CustomInception, ModelArchitecture, CustomResNet
from distortions.dataset.dataloader2 import TrainLoader, get_train_data_loader 
from distortions.utils.functions import mkdir_savel_folder, train_epoch, validate_epoch

def main(model, backbone_name: str, loader: TrainLoader, device: torch.device, num_epochs: int, lr: float, wandb_enable: bool) -> str:
    best_acc = 0.0
    best_loss = 100.00
    model_path = ""

    criterion = nn.CrossEntropyLoss(weight=loader.class_weights.to(device))
    
    # weight_decay para regularização
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    wandb.init(
        mode="online" if wandb_enable else "offline",
        project="distortions-detect",
        config={
            "epochs": num_epochs,
            "learning_rate": lr,
            "batch_size": loader.dataloaders['train'].batch_size,
            "loss_weights": loader.class_weights.tolist(),
            "dataset": loader.dataset_name,
            "train_samples": len(loader.train_dataset),
            "val_samples": len(loader.val_dataset),
            "train_distribution": loader.train_distribution,
            "val_distribution": loader.val_distribution,
            "optimizer": optimizer.__class__.__name__,
            "weight_decay": optimizer.param_groups[0]['weight_decay']
        },
        name=backbone_name
    )

    save_dir = mkdir_savel_folder("runs/train", backbone_name)

    # Save yaml config
    with open(f"{save_dir}/config.yaml", "w") as f:
        for key, value in wandb.config.items():
            f.write(f"{key}: {value}\n")
    
    # Training Loop
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss, train_acc, train_prec, train_rec = train_epoch(model, loader.dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec = validate_epoch(model, loader.dataloaders['val'], criterion, device)

        print(f" ➤ Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% || Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if wandb_enable:
            wandb.log({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "train_precision": train_prec, "train_recall": train_rec,
                "val_precision": val_prec, "val_recall": val_rec
            })

        # Save Best Model
        if val_acc > best_acc and val_loss < best_loss:
            best_acc = val_acc
            best_loss = val_loss
            best_path = f"{save_dir}/best_model_{epoch+1}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"    --> Model Saved! Best Acc: {best_acc:.2f}%")
        
        model_path = f"{save_dir}/last_model.pth"
        torch.save(model.state_dict(), model_path)

        scheduler.step(val_loss)

    if wandb_enable: wandb.finish()
    return best_path

def train_model(
    backbone: ModelArchitecture,
    data_dir: str,
    batch_size: int,
    lr: float,
    num_epochs: int,
    wandb_enable: bool,
    input_size: int = 299,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data + Weights for Loss
    input_size = 299 if backbone == ModelArchitecture.INCEPTION_V3 else 224
    loader = get_train_data_loader(data_dir, batch_size, input_size=input_size)
    
    print(f"Class: {loader.class_names}")
    print(f"Weights to balance: {loader.class_weights}")

    # 2. Initialize Model

    if backbone == ModelArchitecture.INCEPTION_V3:
        model = CustomInception(num_classes=len(loader.class_names), pre_trained=True, training=True)
    elif backbone == ModelArchitecture.NOISE_NET:
        model = NoiseClassificationNet(num_classes=len(loader.class_names))
    else:
        model = CustomResNet(num_classes=len(loader.class_names), backbone=backbone, pretrained=True)
    model = model.to(device)

    return main(model, backbone.value, loader, device, num_epochs, lr, wandb_enable)