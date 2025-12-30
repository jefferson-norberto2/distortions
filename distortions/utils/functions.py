from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch import no_grad
from os import listdir, makedirs

def mkdir_savel_folder(root_path: str, backbone_name: str) -> str:
    # List all folders in the root_path that match the backbone_name pattern
    makedirs(root_path, exist_ok=True)
    
    all_folders = listdir(root_path)

    # count existing runs for the given backbone_name
    run_count = sum(1 for folder in all_folders if folder.startswith(backbone_name))
    new_folder_name = f"{root_path}/{backbone_name}_{run_count + 1}"

    makedirs(new_folder_name, exist_ok=True)
    
    return new_folder_name


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

        # Lógica para Inception (Tuplas) vs Modelos Padrão
        # Nota: Sua NoiseClassificationNet retorna tensor único, então cairá no 'else'
        if isinstance(outputs, (tuple, list)):
            logits, aux_logits = outputs
            loss_main = criterion(logits, labels)
            loss_aux = criterion(aux_logits, labels) * 0.4
            loss = loss_main + loss_aux
            preds_for_metrics = logits
        else:
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

        # Atualizar barra de progresso com cálculo aritmético simples (O(1))
        current_loss = running_loss / total_samples
        current_acc = (correct_predictions / total_samples) * 100
        
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    # --- Cálculo Pesado (Fora do Loop - Executa 1x por época) ---
    # Agora sim usamos o Scikit-Learn
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    return epoch_loss, epoch_acc, precision, recall

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    
    # Acumuladores básicos (RÁPIDOS)
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Listas para métricas completas (usadas só no final)
    all_preds, all_labels = [], []

    with no_grad():
        progress_bar = tqdm(val_loader, desc="Validando", leave=False, dynamic_ncols=True)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Nota: Inception3 em modo .eval() retorna tensor único, 
            # não precisa da lógica de [0] ou tupla aqui.
            
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

            # Atualiza barra com cálculo aritmético O(1)
            current_loss = running_loss / total_samples
            current_acc = (correct_predictions / total_samples) * 100
            
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    # --- Cálculo Pesado (Fora do Loop - Executa 1x por época) ---
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    return val_loss, val_acc, precision, recall
