import torch
import numpy as np

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
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

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100. * val_correct / val_total
    return val_loss, val_acc


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