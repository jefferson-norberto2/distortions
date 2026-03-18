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
# 1. LOGGER CUSTOMIZADO (Estilo YOLO)
# ==========================================
class DualStreamLogger:
    def __init__(self, args, base_dir="runs/dual_stream"):
        self.args = args
        self.save_dir = self._create_folder(base_dir)
        self.results_file = os.path.join(self.save_dir, "results.csv")
        self.history = []
        self._save_yaml()
        print(f"🚀 Experimento iniciado em: {self.save_dir}")

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
# 2. DATASET E MODELO
# ==========================================
class DualDomainDataset(Dataset):
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
            return self.transform(img_rgb), self.transform(img_hsv), label
        return img_rgb, img_hsv, label

class YOLO26DualStream(nn.Module):
    def __init__(self, model_path, num_classes, imgsz=512):
        super().__init__()
        base_rgb = YOLO(model_path).model
        base_hsv = YOLO(model_path).model
        
        # 1. Backbones Independentes
        self.backbone_rgb = base_rgb.model[:-1]
        self.backbone_hsv = base_hsv.model[:-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        with torch.no_grad():
            feat_dim = self.pool(self.backbone_rgb(torch.randn(1, 3, imgsz, imgsz))).shape[1]
        
        total_features = feat_dim * 2
        
        # 2. Módulo de Atenção (Squeeze-and-Excitation Style)
        # O fator de redução (reduction=4) cria um "gargalo" que força a rede
        # a aprender as correlações mais importantes entre os canais.
        reduction_ratio = 4
        self.attention = nn.Sequential(
            nn.Linear(total_features, total_features // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(total_features // reduction_ratio, total_features),
            nn.Sigmoid() # Escala os pesos para o intervalo [0, 1]
        )
        
        # 3. Classificador Final
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_rgb, x_hsv):
        # Extração de características espaciais e de frequência/cor
        f_rgb = self.pool(self.backbone_rgb(x_rgb)).view(x_rgb.size(0), -1)
        f_hsv = self.pool(self.backbone_hsv(x_hsv)).view(x_hsv.size(0), -1)
        
        # Concatenação base
        combined = torch.cat((f_rgb, f_hsv), dim=1)
        
        # Cálculo dos pesos de atenção dinâmicos
        attention_weights = self.attention(combined)
        
        # Recalibração: As características originais são multiplicadas pelos seus pesos
        combined_attended = combined * attention_weights
        
        # Passagem para a classificação final
        return self.classifier(combined_attended)

# ==========================================
# 3. LOOP PRINCIPAL
# ==========================================
def train():
    args = {'imgsz': 512, 'batch': 32, 'epochs': 30, 'lr': 1e-5, 'model': 'yolo26m'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = DualStreamLogger(args)
    
    transform = transforms.Compose([
        transforms.Resize((args['imgsz'], args['imgsz'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = DualDomainDataset('Datasets/CIST/train', transform=transform)
    val_ds = DualDomainDataset('Datasets/CIST/val', transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args['batch'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args['batch'], shuffle=False, num_workers=4)
    model = YOLO26DualStream('yolo26m-cls.pt', len(train_ds.class_names), args['imgsz']).to(device)
    
    # 1. Separação dos parâmetros
    backbone_params = list(model.backbone_rgb.parameters()) + list(model.backbone_hsv.parameters())
    head_params = list(model.attention.parameters()) + list(model.classifier.parameters())

    # 2. Otimizador com Differential LRs
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=1e-2)

    # 3. Scheduler com limite mínimo (eta_min)
    # T_max é o número total de épocas. eta_min garante que a rede nunca pare 100% de aprender.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args['epochs'], 
        eta_min=1e-6 
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0
    for epoch in range(args['epochs']):
        model.train()
        t_loss, t_acc, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}")
        
        for x_rgb, x_hsv, labels in pbar:
            x_rgb, x_hsv, labels = x_rgb.to(device), x_hsv.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(x_rgb, x_hsv)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item() * x_rgb.size(0)
            t_acc += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}, GPU Memory: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

        # Validação
        model.eval()
        v_loss, v_acc, v_total = 0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc="Validing: ")
            for x_rgb, x_hsv, labels in pbar_val:
                x_rgb, x_hsv, labels = x_rgb.to(device), x_hsv.to(device), labels.to(device)
                outputs = model(x_rgb, x_hsv)
                v_loss += criterion(outputs, labels).item() * x_rgb.size(0)
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