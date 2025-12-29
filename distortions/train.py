import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import time
import os

# IMPORTANTE: Importe o novo modelo e o novo loader
from distortions.model.noise_model import NoiseClassificationNet
# Supondo que você salvou o get_data_loaders da minha resposta anterior em algum lugar
from distortions.dataset.dataloader2 import get_data_loaders 
from distortions.utils.functions import train_epoch, validate_epoch

def main(model, backbone_name, dataset_name, train_loader, val_loader, class_weights, device, num_epochs, lr, wandb_enable) -> str:
    best_acc = 0.0
    best_loss = 100.00
    model_path = ""

    # --- MUDANÇA CRÍTICA: Uso de Pesos na Loss ---
    # Isso resolve o desbalanceamento das 600 imagens originais vs 2400 ruídos
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Otimizador: Adicionei weight_decay para regularização
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # WandB Config (Simplificado)
    if wandb_enable:
        wandb.init(
            project="distortions-detect",
            config={
                "architecture": backbone_name,
                "epochs": num_epochs,
                "learning_rate": lr,
                "batch_size": train_loader.batch_size,
                "loss_weights": class_weights.tolist(),
                "dataset": dataset_name,
            },
            name=f"NoiseNet_{backbone_name}"
        )

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = f"runs/train/{backbone_name}_{time_stamp}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        # Assumindo que suas funções train_epoch/validate_epoch já existem e funcionam
        train_loss, train_acc, train_prec, train_rec = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec = validate_epoch(model, val_loader, criterion, device)

        print(f" ➤ Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% || Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if wandb_enable:
            wandb.log({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"]
            })

        # Save Best Model
        if val_acc > best_acc and val_loss < best_loss:
            best_acc = val_acc
            best_loss = val_loss
            model_path = f"{save_dir}/best_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"    --> Model Saved! Best Acc: {best_acc:.2f}%")

        # Scheduler simples (Decaimento a cada 5 épocas)
        if (epoch + 1) % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    if wandb_enable: wandb.finish()
    return model_path

def train_model(
    backbone='resnet_50', # String simples agora
    data_dir="/home/jmn/host/dev/Datasets/IQA/ECSIQ/",
    batch_size=32,
    lr=1e-4,
    num_epochs=16,
    wandb_enable=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Carregar Dados + Pesos (Usando a função da resposta anterior)
    # Nota: Input 299 se for Inception, senão 224 é mais rápido pra Resnet
    input_sz = 299 if 'inception' in backbone else 224
    dataloaders, sizes, class_names, class_weights = get_data_loaders(data_dir, batch_size, input_size=input_sz)
    
    print(f"Classes: {class_names}")
    print(f"Pesos de Balanceamento: {class_weights}")

    # 2. Instanciar o Modelo Especializado
    # Não usamos mais get_backbone_and_weights isolado
    model = NoiseClassificationNet(num_classes=len(class_names))
    model = model.to(device)

    dataset_name = data_dir.strip('/').split('/')[-1]

    return main(model, backbone, dataset_name, dataloaders['train'], dataloaders['val'], class_weights, device, num_epochs, lr, wandb_enable)

# --- Execução ---
if __name__ == "__main__":
    # Exemplo de chamada
    backbone_choice = 'resnet_50' # ou 'inception_v3'
    best_model_path = train_model(
        backbone=backbone_choice, 
        data_dir='./data/ECSIQ/', # Ajuste seu caminho
        num_epochs=20, 
        batch_size=16, # Batch menor para Inception/ResNet50 caber na VRAM
        wandb_enable=True
    )