import matplotlib.pyplot as plt
import torch
import wandb 
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from distortions.dataset.dataloaders import get_val_dataloader
from distortions.dataset.dataloaders import get_val_dataloader
from distortions.model.custom_resnet import CustomResNet
from distortions.model.custom_inception import CustomInception
from distortions.utils.functions import validate_epoch  
from distortions.model.custom_mobilenet import CustomMobileNetV3
from distortions.model.distortion_hunter import DistortionHunter


def test_model(folder_path="/home/jmn/host/dev/Datasets/IQA/ELIVE/",
               weight_path="distortions_10_resnet50_b16_lr1e-4.pth", 
               name_model="resnet50", wandb_enable=False,
               ):
    
    # --- Inicializa o W&B ---
    wandb.init(project="distortions-detect", name=f"evaluation_{name_model}", mode="online" if wandb_enable else "disabled")

    # --- Dataset e DataLoader ---
    val_loader, dataset = get_val_dataloader(data_dir=folder_path, batch_size=16, im_size=300 if name_model == 'inception_v3' else 224)
    class_names = dataset.classes

    # --- Dispositivo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if name_model == 'inception_v3':
        model = CustomInception(num_classes=len(dataset.classes), pre_treined=False, training=False)
    elif name_model.startswith("mobilenet_v3"):
        model = CustomMobileNetV3(num_classes=len(dataset.classes), pre_trained=False, backbone=name_model)
    elif name_model.startswith("distortion_hunter"):
        model = DistortionHunter(num_classes=len(dataset.classes))
    else:
        model = CustomResNet(num_classes=len(dataset.classes), pre_treined=False, backbone=name_model)
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    # --- Avaliação ---
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc, precision, recall, all_preds, all_labels = validate_epoch(model, val_loader, criterion, device)
    print(f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f} |")
    wandb.log({"val_loss": val_loss, "val_acc": val_acc, "val_precision": precision, "val_recall": recall})

    # --- Cria a matriz de confusão ---
    cm = confusion_matrix(all_preds, all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    base_dir = "runs/test"
    os.makedirs(base_dir, exist_ok=True)
    
    count = sum(1 for folder in os.listdir(base_dir) if folder.startswith(name_model))
            
    save_dir = f"{base_dir}/{name_model}_{count+1}"
    os.makedirs(save_dir, exist_ok=True)

    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel('True Label')     
    plt.ylabel('Predicted Label')
    plt.savefig(f"{save_dir}/confusion_matrix_{name_model}.png", dpi=300)
    wandb.log({f"confusion_matrix_{name_model}": wandb.Image(f"{save_dir}/confusion_matrix_{name_model}.png")})

if __name__ == '__main__':
    test_model()
