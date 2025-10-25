import torch
import numpy as np
from torchvision import models

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

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from distortions.model.custom_resnet import CustomResNet
from distortions.utils.functions import get_backbone_and_weights
from distortions.dataset.dataset import get_dataloaders

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # tqdm cria a barra de progresso
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
        correct += predicted.eq(labels).sum().item()

        # Atualiza o texto mostrado na barra
        current_loss = running_loss / total
        current_acc = 100. * correct / total
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validando", leave=False, dynamic_ncols=True)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            current_loss = running_loss / total
            current_acc = 100. * correct / total
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = 100. * correct / total
    return val_loss, val_acc
