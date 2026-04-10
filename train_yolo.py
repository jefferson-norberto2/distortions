import torch
import kornia
import types
import math
from ultralytics import YOLO

def inject_lab(trainer_or_validator):
    if getattr(trainer_or_validator, "lab_patch_applied", False):
        return
    
    trainer_or_validator.lab_patch_applied = True

    original_preprocess_batch = trainer_or_validator.preprocess_batch

    def custom_preprocess_batch(self, batch):
        batch = original_preprocess_batch(batch)
        
        imagens_rgb = batch['img']
        
        imagens_lab = kornia.color.rgb_to_lab(imagens_rgb)
        
        L = imagens_lab[:, 0:1, :, :] / 100.0
        a = (imagens_lab[:, 1:2, :, :] + 128.0) / 255.0
        b = (imagens_lab[:, 2:3, :, :] + 128.0) / 255.0
        
        batch['img'] = torch.cat([L, a, b], dim=1)
        
        return batch

    trainer_or_validator.preprocess_batch = types.MethodType(custom_preprocess_batch, trainer_or_validator)

def inject_hsv(trainer_or_validator):
    if getattr(trainer_or_validator, "hsv_patch_applied", False):
        return
    trainer_or_validator.hsv_patch_applied = True

    original_preprocess_batch = trainer_or_validator.preprocess_batch

    def custom_preprocess_batch(self, batch):
        batch = original_preprocess_batch(batch)
        imagens_rgb = batch['img']
        imagens_hsv = kornia.color.rgb_to_hsv(imagens_rgb)
        
        # 3. Normalização
        H = imagens_hsv[:, 0:1, :, :] / (2 * math.pi)
        S = imagens_hsv[:, 1:2, :, :]
        V = imagens_hsv[:, 2:3, :, :]
        
        batch['img'] = torch.cat([H, S, V], dim=1)
        
        return batch

    trainer_or_validator.preprocess_batch = types.MethodType(custom_preprocess_batch, trainer_or_validator)


weights = [
    'yolo26n-cls.pt',
    'yolo26s-cls.pt',
    'yolo26m-cls.pt',
    'yolo26l-cls.pt',
    'yolo26x-cls.pt'
]

colors_space = {
    'lab': inject_lab,
    'hsv': inject_hsv
}

for key, inject_function in colors_space.items():

    for weight in weights:

        model = YOLO(weight)  

        model.add_callback("on_train_batch_start", inject_function)
        model.add_callback("on_val_batch_start", inject_function)

        model.train(
            data='Datasets/LIST', 
            task=f'classify',
            project=f'{weight.split("-")[0]}_{key}',
            name=weight.split('-')[1].split('.')[0],
            epochs=30, 
            imgsz=512, 
            device=0, 
            batch=32,
            # --- Desativando Augmentations ---
            hsv_h=0.0,      
            hsv_s=0.0,      
            hsv_v=0.0,      
            degrees=0.0,    
            translate=0.0,  
            scale=0.0,      
            shear=0.0,      
            perspective=0.0,
            flipud=0.0,     
            fliplr=0.5,     
            mosaic=0.0,     
            mixup=0.0,      
            auto_augment=None, 
            erasing=0.0     
        )