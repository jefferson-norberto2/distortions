import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

from distortions.utils.dual_logger import DualLogger
from distortions.dataset.dual_dataset import DualDataset
from distortions.model.early_fusion import EarlyFusionAdapter 

FAMILY, VERSION = "early_fusion", "v1"

def train(args: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_path = Path(args['dataset_path']) / 'train'
    val_path = Path(args['dataset_path']) / 'val'
    train_ds = DualDataset(train_path, image_mode=args['image_mode'], image_size=(args['imgsz'], args['imgsz']))
    val_ds = DualDataset(val_path, image_mode=args['image_mode'], image_size=(args['imgsz'], args['imgsz']))
    
    # Added drop_last=True to prevent BatchNorm crashing on uneven batches
    train_loader = DataLoader(train_ds, batch_size=args['batch'], shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args['batch'], shuffle=False, num_workers=4)
    
    # 2. Model initialization now only requires one backbone
    model = EarlyFusionAdapter(len(train_ds.class_names)).to(device)

    logger = DualLogger(args, base_dir=f"runs/trained/{FAMILY}/{VERSION}", train_dataset=train_ds, val_dataset=val_ds)
    
    # 3. Parameter separation
    optimizer = optim.AdamW(model.parameters(), lr=args['lr_backbone'], weight_decay=1e-2)

    # T_max is the total number of epochs. eta_min ensures the network never completely stops learning.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args['epochs'], 
        eta_min=1e-6 
    )
    
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(args['epochs']):
        # ================== TRAINING ==================
        model.train()
        t_loss, t_acc, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}")
        
        for x_rgb, x_hsv, labels in pbar:
            x_rgb, x_hsv, labels = x_rgb.to(device), x_hsv.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # The model automatically concatenates the tensors internally now
            outputs = model(x_rgb, x_hsv)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in the early epochs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            t_loss += loss.item() * x_rgb.size(0)
            t_acc += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            
            mem_info = f", VRAM: {torch.cuda.memory_reserved(device) / 1024**3:.2f}GB" if device_type == 'cuda' else ""
            pbar.set_postfix(loss=f"{loss.item():.4f}{mem_info}")

        # ================== VALIDATION ==================
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

        # ================== METRICS & SAVING ==================
        scheduler.step()
        logger.log_epoch(epoch, t_loss/total, v_loss/v_total, t_acc/total, v_acc/v_total)
        
        if (v_acc/v_total) > best_acc:
            best_acc = v_acc/v_total
            # If you add the .save() method to EarlyFusionAdapter later, you can replace this line
            torch.save(model.state_dict(), logger.save_dir / "best.pt")
            logger.save_final_metrics(y_true, y_pred, train_ds.class_names)
            print(f"🌟 New best model! Accuracy: {best_acc:.4f}")

def train_early_fusion():
    args = {
        'imgsz': 512, 
        'batch': 32, 
        'epochs': 40, 
        'lr_backbone': 1e-5, 
        'lr_classifier': 1e-5, 
        'backbone': 'mobilenet_v1',
        'dataset_path': '/run/media/jmn/Removable Disk/Datasets/LIST',
        'image_mode': 'HSV'
    }
    train(args)