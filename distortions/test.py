import matplotlib.pyplot as plt
import torch
import wandb 
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from distortions.utils.functions import validate_epoch, mkdir_savel_folder
from distortions.model.noise_model import NoiseClassificationNet
from distortions.model.custom_resnet import ModelArchitecture, CustomInception, CustomResNet
from distortions.dataset.dataloader2 import get_test_data_loader

def test_model(data_dir: str, model_path: str, backbone: ModelArchitecture, wandb_enable: bool, batch_size: int):
    # --- Salvamento da matriz de confusão ---
    save_dir = mkdir_savel_folder("runs/test", backbone.value)
    
    # --- Inicializa o W&B ---
    wandb.init(project="distortions-detect", name=f"evaluation_{backbone.value}", mode="online" if wandb_enable else "disabled")
    
   # --- Carrega os dados de validação ---
    input_size = 299 if backbone == ModelArchitecture.INCEPTION_V3 else 224
    loader = get_test_data_loader(data_dir, batch_size=batch_size, input_size=input_size)

    # --- Dispositivo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Modelo -- 
    if backbone == ModelArchitecture.NOISE_NET:
        model = NoiseClassificationNet(num_classes=len(loader.class_names))
    elif backbone == ModelArchitecture.INCEPTION_V3:
        model = CustomInception(num_classes=len(loader.class_names), pre_trained=False, training=False)
    else:
        model = CustomResNet(num_classes=len(loader.class_names), backbone=backbone, pretrained=False)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # --- Avaliação ---
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc, precision, recall = validate_epoch(model, loader.dataloader, criterion, device)
    print(f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f} |")
    wandb.log({"val_loss": val_loss, "val_acc": val_acc, "val_precision": precision, "val_recall": recall})

    # --- Geração da matriz de confusão ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader.dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Matriz de confusão ---
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=loader.class_names)


    # --- Visualização ---
    _, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    plt.title(f"Matriz de Confusão - {backbone}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix_{backbone}.png", dpi=300)
    wandb.log({f"confusion_matrix_{backbone}": wandb.Image(f"{save_dir}/confusion_matrix_{backbone}.png")})

if __name__ == '__main__':
    test_model()
