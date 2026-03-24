import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. LOGGER CUSTOMIZADO
# ==========================================
class EarlyFusionLogger:
    def __init__(self, args, base_dir="runs/early_fusion"):
        self.args = args
        self.save_dir = self._create_folder(base_dir)
        self.results_file = os.path.join(self.save_dir, "results.csv")
        self.history = []
        self._save_yaml()
        print(f"🚀 Experimento Early Fusion iniciado em: {self.save_dir}")

    def _create_folder(self, base_dir):
        os.makedirs(base_dir, exist_ok=True)
        existing = [d for d in os.listdir(base_dir) if d.startswith("train")]
        index = len(existing) + 1
        save_dir = os.path.join(base_dir, f"train{index}")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _save_yaml(self):
        with open(os.path.join(self.save_dir, "args.yaml"), 'w') as f:
            yaml.dump(self.args, f, default_flow_style=False)

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc):
        row = {"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss, 
               "metrics/accuracy": val_acc, "train/accuracy": train_acc}
        self.history.append(row)
        pd.DataFrame(self.history).to_csv(self.results_file, index=False)
        self._plot_results()

    def _plot_results(self):
        df = pd.read_csv(self.results_file)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(df['epoch'], df['train/loss'], label='train')
        ax[0].plot(df['epoch'], df['val/loss'], label='val')
        ax[0].set_title('Loss'); ax[0].legend()
        ax[1].plot(df['epoch'], df['metrics/accuracy'], label='val_acc')
        ax[1].set_title('Accuracy'); ax[1].legend()
        plt.savefig(os.path.join(self.save_dir, "results.png"), dpi=300)
        plt.close()

    def save_final_metrics(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
        for matrix, name in zip([cm, cm_norm], ["confusion_matrix.png", "confusion_matrix_normalized.png"]):
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, fmt='.2f' if 'norm' in name else 'd', 
                        cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted'); plt.ylabel('True')
            plt.savefig(os.path.join(self.save_dir, name), dpi=300); plt.close()

# ==========================================
# 2. DATASET E MODELO (6 CANAIS)
# ==========================================
class EarlyFusionDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.class_names = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        for cls_name in self.class_names:
            cls_path = Path(root) / cls_name
            for img_path in cls_path.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_rgb = Image.open(path).convert('RGB')
        img_hsv = img_rgb.convert('HSV')
        
        if self.transform:
            t_rgb = self.transform(img_rgb)
            t_hsv = self.transform(img_hsv)
            # Concatena os canais para formar um tensor [6, H, W]
            img_combined = torch.cat((t_rgb, t_hsv), dim=0)
            return img_combined, label
            
        return img_rgb, img_hsv, label

class YOLO26EarlyFusion(nn.Module):
    def __init__(self, model_path, num_classes):
        super().__init__()
        
        # Carrega o modelo COMPLETO (sem o [:-1])
        self.model = YOLO(model_path).model
        
        # ==========================================
        # 1. Modifica a ENTRADA (Primeira Camada)
        # ==========================================
        first_layer = self.model.model[0].conv
        new_conv = nn.Conv2d(
            in_channels=6, 
            out_channels=first_layer.out_channels, 
            kernel_size=first_layer.kernel_size, 
            stride=first_layer.stride, 
            padding=first_layer.padding, 
            bias=False
        )
        
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = first_layer.weight.clone()
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            
        self.model.model[0].conv = new_conv
        
        # ==========================================
        # 2. Modifica a SAÍDA (Classificador Original)
        # ==========================================
        # A última camada do YOLO de classificação é o módulo 'Classify' (índice -1)
        # Dentro dele, existe a camada linear final que precisamos ajustar para as suas classes
        in_features = self.model.model[-1].linear.in_features
        self.model.model[-1].linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Apenas repassa para o modelo nativo, simples assim!
        return self.model(x)

# ==========================================
# 3. LOOP PRINCIPAL
# ==========================================
def train():
    args = {'imgsz': 512, 'batch': 16, 'epochs': 30, 'lr': 1e-3, 'model': 'yolo26m'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = EarlyFusionLogger(args)
    
    transform = transforms.Compose([
        transforms.Resize((args['imgsz'], args['imgsz'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = EarlyFusionDataset('Datasets/CIST/train', transform=transform)
    val_ds = EarlyFusionDataset('Datasets/CIST/val', transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args['batch'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args['batch'], shuffle=False, num_workers=4)
    
    model = YOLO26EarlyFusion('yolo26m-cls.pt', len(train_ds.class_names)).to(device)
    
    # 1. Separação dos parâmetros da nova arquitetura nativa
    head_params = list(model.model.model[-1].linear.parameters())
    head_param_ids = set(id(p) for p in head_params)
    
    # Tudo que não for a camada linear final, consideramos como "backbone"
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    # 2. Otimizador com Differential LRs
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0
    for epoch in range(args['epochs']):
        model.train()
        t_loss, t_acc, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}")
        
        for x_combined, labels in pbar:
            # Envia o tensor único de 6 canais para a GPU
            x_combined, labels = x_combined.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(x_combined)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item() * x_combined.size(0)
            t_acc += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", vram=f"{torch.cuda.memory_reserved(device) / 1024**3:.2f}GB")

        # Validação
        model.eval()
        v_loss, v_acc, v_total = 0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc="Validating: ")
            for x_combined, labels in pbar_val:
                x_combined, labels = x_combined.to(device), labels.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(x_combined)
                    
                    # --- A MÁGICA PARA O EVAL DO YOLO ---
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0] # Pega apenas o tensor de logits
                    # ------------------------------------
                    
                    loss = criterion(outputs, labels)
                
                v_loss += loss.item() * x_combined.size(0)
                preds = outputs.argmax(1)
                v_acc += (preds == labels).sum().item()
                v_total += labels.size(0)
                y_true.extend(labels.cpu().numpy()); y_pred.extend(preds.cpu().numpy())

        scheduler.step()
        logger.log_epoch(epoch, t_loss/total, v_loss/v_total, t_acc/total, v_acc/v_total)
        
        if (v_acc/v_total) > best_acc:
            best_acc = v_acc/v_total
            torch.save(model.state_dict(), os.path.join(logger.save_dir, "best.pt"))
            logger.save_final_metrics(y_true, y_pred, train_ds.class_names)

if __name__ == "__main__":
    train()