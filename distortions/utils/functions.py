from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
import torch



def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    
    # Acumuladores básicos (RÁPIDOS)
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Listas para métricas completas (usadas só no final)
    all_preds, all_labels = [], []

    progress_bar = tqdm(train_loader, desc="Treinando", leave=False, dynamic_ncols=True)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(images)

        # Checa se o modelo retornou a estrutura do Inception no modo treino
        if hasattr(outputs, 'logits') and hasattr(outputs, 'aux_logits'):
            loss_main = criterion(outputs.logits, labels)
            loss_aux = criterion(outputs.aux_logits, labels) * 0.4
            loss = loss_main + loss_aux
            preds_for_metrics = outputs.logits
        else:
            # Para ResNet, EfficientNet, ou se aux_logits estiver desativado
            loss = criterion(outputs, labels)
            preds_for_metrics = outputs

        # Backward Pass
        loss.backward()
        optimizer.step()

        # --- Métricas "On-the-fly" (CPU Leve) ---
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        
        _, predicted = preds_for_metrics.max(1)
        correct_predictions += predicted.eq(labels).sum().item()
        total_samples += batch_size

        # Guardar para o final (sem calcular sklearn aqui)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        current_loss = running_loss / total_samples
        current_acc = accuracy_score(all_labels, all_preds) * 100
        
        gpu_mem = 0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_reserved(device) / 1024**3

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}", 
            acc=f"{current_acc:.2f}%", 
            mem=f"{gpu_mem:.2f}GB"
        )

    train_loss = running_loss / len(train_loader.dataset)
    
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    
    return train_loss, current_acc, precision, recall

    return epoch_loss, epoch_acc, precision, recall

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    
    # Acumuladores básicos (RÁPIDOS)
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Listas para métricas completas (usadas só no final)
    all_preds, all_labels = [], []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validando", leave=False, dynamic_ncols=True)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Acumuladores de Loss e Acurácia Simples
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(labels).sum().item()
            total_samples += batch_size

            # Armazena para métricas finais
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_loss = running_loss / total_samples
            current_acc = accuracy_score(all_labels, all_preds) * 100
            
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    # --- Cálculo Pesado (Fora do Loop - Executa 1x por época) ---
    val_loss = running_loss / len(val_loader.dataset)
    
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    
    return val_loss, current_acc, precision, recall, all_preds, all_labels
