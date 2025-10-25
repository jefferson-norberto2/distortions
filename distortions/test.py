import matplotlib.pyplot as plt
import torch
import wandb 
import time
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from distortions.model.custom_resnet import CustomResNet
from distortions.utils.functions import get_backbone_and_weights, validate_epoch  # se quiser manter o uso atual

def test_model(folder_path="/home/jmn/host/dev/Datasets/IQA/ELIVE/",
               weight_path="distortions_10_resnet50_b16_lr1e-4.pth", name_model="resnet50"):
    
    # --- Inicializa o W&B ---
    wandb.init(project="distortions-detect", name=f"evaluation_{name_model}")
    
    # --- Transformação das imagens ---
    transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
    ])

    # --- Dataset e DataLoader ---
    dataset = datasets.ImageFolder(root=folder_path, transform=transform)
    val_loader = DataLoader(dataset, batch_size=20, shuffle=False)
    class_names = dataset.classes  # nomes das classes

    # --- Dispositivo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Modelo ---
    backbone, weights = get_backbone_and_weights(name_model=name_model)
    
    model = CustomResNet(num_classes=7, backbone=backbone, weights=weights)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    # --- Avaliação ---
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc, precision, recall = validate_epoch(model, val_loader, criterion, device)
    print(f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f} |")
    wandb.log({"val_loss": val_loss, "val_acc": val_acc, "val_precision": precision, "val_recall": recall})

    # --- Geração da matriz de confusão ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Cria a matriz de confusão ---
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = f"runs/test/{time_stamp}"
    os.makedirs(save_dir, exist_ok=True)

    # --- Visualização ---
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    plt.title(f"Matriz de Confusão - {name_model}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix_{name_model}.png", dpi=300)
    wandb.log({f"confusion_matrix_{name_model}": wandb.Image(f"{save_dir}/confusion_matrix_{name_model}.png")})

if __name__ == '__main__':
    test_model()
