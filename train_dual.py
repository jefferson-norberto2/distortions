import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from distortions.utils.dual_logger import DualLogger
from distortions.dataset.dual_dataset import DualDataset
from distortions.model.dual_stream import DualStream
from pathlib import Path

FAMILY, VERSION = "dual_stream", "v1"

def train(args: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_path = Path(args['dataset_path']) / 'train'
    val_path = Path(args['dataset_path']) / 'val'
    train_ds = DualDataset(train_path, image_mode=args['image_mode'])
    val_ds = DualDataset(val_path, image_mode=args['image_mode'])
    train_loader = DataLoader(train_ds, batch_size=args['batch'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args['batch'], shuffle=False, num_workers=4)
    model = DualStream(args['rgb_head'], args['hsv_head'], len(train_ds.class_names)).to(device)

    logger = DualLogger(args, base_dir=f"runs/trained/{FAMILY}/{VERSION}/train_1", train_dataset=train_ds, val_dataset=val_ds)
    
    backbone_params = list(model.rgb_head.parameters()) + list(model.hsv_head.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args['lr_backbone']},
        {'params': classifier_params, 'lr': args['lr_classifier']}
    ], weight_decay=1e-2)

    # T_max é o número total de épocas. eta_min garante que a rede nunca pare 100% de aprender.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args['epochs'], 
        eta_min=1e-6 
    )
    
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(args['epochs']):
        # ================== TREINO ==================
        model.train()
        t_loss, t_acc, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}")
        
        for x_rgb, x_hsv, labels in pbar:
            x_rgb, x_hsv, labels = x_rgb.to(device), x_hsv.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(x_rgb, x_hsv)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()

            t_loss += loss.item() * x_rgb.size(0)
            t_acc += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            
            mem_info = f", VRAM: {torch.cuda.memory_reserved(device) / 1024**3:.2f}GB" if device_type == 'cuda' else ""
            pbar.set_postfix(loss=f"{loss.item():.4f}{mem_info}")

        # ================== VALIDAÇÃO ==================
        model.eval()
        v_loss, v_acc, v_total = 0, 0, 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc="Validating: ")
            for x_rgb, x_hsv, labels in pbar_val:
                x_rgb, x_hsv, labels = x_rgb.to(device), x_hsv.to(device), labels.to(device)
                
                outputs = model(x_rgb, x_hsv)
                loss = criterion(outputs, labels)
                
                v_loss += loss.item() * x_rgb.size(0)
                preds = outputs.argmax(1)
                v_acc += (preds == labels).sum().item()
                v_total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # ================== MÉTRICAS E SALVAMENTO ==================
        scheduler.step()
        logger.log_epoch(epoch, t_loss/total, v_loss/v_total, t_acc/total, v_acc/v_total)
        
        if (v_acc/v_total) > best_acc:
            best_acc = v_acc/v_total
            torch.save(model.state_dict(), logger.save_dir / "best.pt")
            logger.save_final_metrics(y_true, y_pred, train_ds.class_names)
            print(f"🌟 New best model! Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    args = {'imgsz': 512, 
            'batch': 32, 
            'epochs': 30, 
            'lr_backbone': 1e-5, 
            'lr_classifier': 1e-5,
            'rgb_head': 'mobilenet_v1', 
            'hsv_head': 'mobilenet_v1',
            'dataset_path': '/run/media/jmn/Removable Disk/Datasets/LIST',
            'image_mode': 'HSV'}
    train(args)