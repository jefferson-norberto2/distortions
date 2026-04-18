import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from distortions.utils.dual_logger import DualLogger
from distortions.dataset.single_dataset import SingleDataset
from distortions.model.custom_mobilenet import CustomMobileNet
from pathlib import Path


def train(args: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((args['imgsz'], args['imgsz'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_path = Path(args['dataset_path']) / 'train'
    val_path = Path(args['dataset_path']) / 'val'
    train_ds = SingleDataset(train_path, image_mode=args['image_mode'])
    val_ds = SingleDataset(val_path, image_mode=args['image_mode'])
    train_loader = DataLoader(train_ds, batch_size=args['batch'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args['batch'], shuffle=False, num_workers=4)
    model = CustomMobileNet(len(train_ds.class_names), True, args['model']).to(device)
    logger = DualLogger(args, base_dir=f"runs/{args['model']}_{args['image_mode']}", train_dataset=train_ds, val_dataset=val_ds)
    
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0
    for epoch in range(args['epochs']):
        model.train()
        t_loss, t_acc, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}")
        
        for x_rgb, labels in pbar:
            x_rgb, labels = x_rgb.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(x_rgb)
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
            for x_rgb, labels in pbar_val:
                x_rgb, labels = x_rgb.to(device), labels.to(device)
                outputs = model(x_rgb)
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
    models = ['mobilenet_v2']
    args = {
        'imgsz': 512, 
        'batch': 32, 
        'epochs': 30, 
        'lr': 1e-5, 
        'dataset_path': 'Datasets/LIST',
        'image_mode': 'LAB'
    }
    for model in models:
        args['model'] = model
        train(args)

    args['image_mode'] = 'HSV'
    for model in models:
        args['model'] = model
        train(args)