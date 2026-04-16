import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from distortions.utils.dual_logger import DualLogger
from distortions.dataset.dual_dataset import DualDataset
from distortions.model.dual_stream import DualStream
from pathlib import Path


def train(args: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_path = Path(args['dataset_path']) / 'train'
    val_path = Path(args['dataset_path']) / 'val'
    train_ds = DualDataset(train_path, image_mode=args['image_mode'])
    val_ds = DualDataset(val_path, image_mode=args['image_mode'])
    train_loader = DataLoader(train_ds, batch_size=args['batch'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args['batch'], shuffle=False, num_workers=4)
    model = DualStream(args['model_1'], args['model_2'], len(train_ds.class_names)).to(device)

    logger = DualLogger(args, base_dir="runs/dual_stream_v2", train_dataset=train_ds, val_dataset=val_ds)
    
    # 1. Separação dos parâmetros
    backbone_params = list(model.rgb_head.parameters()) + list(model.hsv_head.parameters())
    classifier_params = list(model.classifier.parameters())

    # 2. Otimizador com Differential LRs
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args['lr_backbone']},
        {'params': classifier_params, 'lr': args['lr_classifier']}
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
            torch.save(model.state_dict(), logger.save_dir / "best.pt")
            logger.save_final_metrics(y_true, y_pred, train_ds.class_names)

if __name__ == "__main__":
    args = {'imgsz': 512, 
            'batch': 32, 
            'epochs': 30, 
            'lr_backbone': 1e-5, 
            'lr_classifier': 1e-5,
            'model_1': 'mobilenet_v3_large', 
            'model_2': 'mobilenet_v3_large',
            'dataset_path': 'Datasets/CIST',
            'image_mode': 'HSV'}
    train(args)