import torch
import numpy as np
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

def get_backbone_and_weights(name_model='resnet_50'):
    if name_model == 'resnet_18':
        back = models.resnet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    if name_model == 'resnet_34':
        back = models.resnet34
        weights = models.ResNet34_Weights.IMAGENET1K_V1
    elif name_model == 'resnet_50':
        back = models.resnet50
    elif name_model == 'resnet_101':
        back = models.resnet101
        weights = models.ResNet101_Weights.IMAGENET1K_V2
    elif name_model == 'resnet_152':
        back = models.resnet152
        weights = models.ResNet152_Weights.IMAGENET1K_V2
    else:
        raise ValueError(f"Modelo desconhecido: {name_model}, escolha entre 'resnet_18', 'resnet_34', 'resnet_50', 'resnet_101' ou 'resnet_152'.")
    return back, weights

def class_distribution(dataset, train_dataset, val_dataset):
    # classes do dataset
    class_names = dataset.classes

    # inicializa contadores
    train_counts = np.zeros(len(class_names), dtype=int)
    val_counts   = np.zeros(len(class_names), dtype=int)

    # contar exemplos no train_dataset
    for idx in train_dataset.indices:
        label = dataset[idx][1]  # índice da classe
        train_counts[label] += 1

    # contar exemplos no val_dataset
    for idx in val_dataset.indices:
        label = dataset[idx][1]
        val_counts[label] += 1

    # imprimir resultados
    print("Distribuição por classe:\n")
    for i, cls in enumerate(class_names):
        print(f"{cls:15s} | Treino: {train_counts[i]} | Validação: {val_counts[i]} | Total: {train_counts[i] + val_counts[i]}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    progress_bar = tqdm(train_loader, desc="Treinando", leave=False, dynamic_ncols=True)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)

        # Armazena para métricas
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Atualiza a barra
        current_loss = running_loss / total
        current_acc = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%", precision=f"{precision:.2f}%", recall=f"{recall:.2f}%")

    # Cálculos finais
    train_loss = running_loss / len(train_loader.dataset)
    
 
    return train_loss, current_acc, precision, recall


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validando", leave=False, dynamic_ncols=True)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)

            # Armazena para métricas
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_loss = running_loss / total
            current_acc = accuracy_score(all_labels, all_preds) * 100
            precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
            recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%", precision=f"{precision:.2f}%", recall=f"{recall:.2f}%")

    val_loss = running_loss / len(val_loader.dataset)
    

    return val_loss, current_acc, precision, recall
