import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from distortions.model.custom_mobilenet import CustomMobileNetV3
from distortions.model.custom_resnet import CustomResNet
from distortions.model.custom_inception import CustomInception
from distortions.utils.functions import train_epoch, validate_epoch
from distortions.dataset.dataloaders import get_train_dataloaders  
from distortions.model.distortion_hunter import DistortionHunter
from distortions.model.resnet18_gdn import resnet18_gdn

def main(model, backbone, dataset_name, train_loader, val_loader, train_dataset, val_dataset, device, num_epochs, lr, wandb_enable):
    best_acc = 0.0
    model_path = ""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    wandb.init(
        mode="online" if wandb_enable else "disabled",
        project="distortions-detect",
        config={
            "architecture": backbone,
            "epochs": num_epochs,
            "learning_rate": lr,
            "batch_size": train_loader.batch_size,
            "optimizer": type(optimizer).__name__,
            "criterion": criterion._get_name(),
            "dataset": dataset_name,
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "num_classes": len(train_dataset.classes),
            "classes": train_dataset.classes,
        },
        name=f"training_{backbone}"
    )
    
    base_dir = "runs/train"
    os.makedirs(base_dir, exist_ok=True)
    
    count = sum(1 for folder in os.listdir(base_dir) if folder.startswith(backbone)) 
            
    save_dir = f"{base_dir}/{backbone}_{count+1}"
    
    while os.path.exists(save_dir):
        count += 1
        save_dir = f"{base_dir}/{backbone}_{count+1}"
    
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/config.yaml", "w") as file:
        for key, value in wandb.config.items():
            file.write(f"{key}: {value}\n")

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss, train_acc, train_precision, train_recall  = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, all_preds, all_labels = validate_epoch(model, val_loader, criterion, device)

        print(f"  ➤ Train Loss: {train_loss:.4f}"
              f"| Val Acc: {val_acc:.2f}%, Vall Precision: {val_precision:.2f}%, Vall Recall: {val_recall:.2f}%")

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

        if val_acc > best_acc:
            best_acc = val_acc
            model_path = f"{save_dir}/best.pth"
            torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        if epoch == num_epochs - 1:            
            cm = confusion_matrix(all_preds, all_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_loader.dataset.classes)
            disp.plot(cmap=plt.cm.Blues)
            plt.xlabel('True Label')     
            plt.ylabel('Predicted Label') 
            plt.savefig(f"{save_dir}/confusion_matrix.png")

            cm_norm = confusion_matrix(all_preds, all_labels, normalize='true')
            disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=train_loader.dataset.classes)
            disp_norm.plot(cmap=plt.cm.Blues)
            plt.xlabel('True Label')
            plt.ylabel('Predicted Label')
            plt.savefig(f"{save_dir}/confusion_matrix_normalized.png")

    wandb.finish()

def train_model(
    backbone='resnet50',
    data_dir="/home/jmn/host/dev/Datasets/IQA/ECSIQ/",
    img_size=320,
    batch_size=32,
    lr=1e-4,
    num_epochs=10,
    wandb_enable=True
):
    train_loader, val_loader, train_dataset, val_dataset = get_train_dataloaders(
        data_dir=data_dir, 
        batch_size=batch_size,
        img_size=img_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if backbone.lower() == "inceptionv3":
        model = CustomInception(
            num_classes=len(train_dataset.classes),
            pre_treined=True,
            training=True
        ).to(device)
    elif backbone.lower().startswith("mobilenet_v3"):
        model = CustomMobileNetV3(
            num_classes=len(train_dataset.classes),
            pre_trained=True,
            backbone=backbone
        ).to(device)
    elif backbone.lower().startswith("distortion_hunter"):
        model = DistortionHunter(
            num_classes=len(train_dataset.classes)
        ).to(device)
    elif backbone.lower().startswith("resnet18gdn"):
        model = resnet18_gdn(num_classes=len(train_dataset.classes)).to(device)
    else:
        model = CustomResNet(
            num_classes=len(train_dataset.classes),
            pre_treined=True,
            backbone=backbone
        ).to(device)

    dataset_name = data_dir.strip('/').split('/')[-1]

    main(model, backbone, dataset_name, train_loader, val_loader, train_dataset, val_dataset, device, num_epochs, lr, wandb_enable)


    