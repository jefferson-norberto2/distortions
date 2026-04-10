import torch
import kornia
import types
import math
from ultralytics import YOLO

def inject_lab(trainer_or_validator):
    if getattr(trainer_or_validator, "lab_patch_applied", False):
        return
    
    trainer_or_validator.lab_patch_applied = True

    # Detect the correct method name based on the object type (Trainer vs Validator)
    method_name = "preprocess_batch" if hasattr(trainer_or_validator, "preprocess_batch") else "preprocess"
    original_preprocess = getattr(trainer_or_validator, method_name)

    def custom_preprocess(self, batch):
        batch = original_preprocess(batch)
        
        images_rgb = batch['img']
        images_lab = kornia.color.rgb_to_lab(images_rgb)
        
        l_channel = images_lab[:, 0:1, :, :] / 100.0
        a_channel = (images_lab[:, 1:2, :, :] + 128.0) / 255.0
        b_channel = (images_lab[:, 2:3, :, :] + 128.0) / 255.0
        
        batch['img'] = torch.cat([l_channel, a_channel, b_channel], dim=1)
        
        return batch

    # Apply the patch to the correct method
    setattr(trainer_or_validator, method_name, types.MethodType(custom_preprocess, trainer_or_validator))

def inject_hsv(trainer_or_validator):
    if getattr(trainer_or_validator, "hsv_patch_applied", False):
        return
    
    trainer_or_validator.hsv_patch_applied = True

    # Detect the correct method name based on the object type (Trainer vs Validator)
    method_name = "preprocess_batch" if hasattr(trainer_or_validator, "preprocess_batch") else "preprocess"
    original_preprocess = getattr(trainer_or_validator, method_name)

    def custom_preprocess(self, batch):
        batch = original_preprocess(batch)
        
        images_rgb = batch['img']
        images_hsv = kornia.color.rgb_to_hsv(images_rgb)
        
        # Normalization
        h_channel = images_hsv[:, 0:1, :, :] / (2 * math.pi)
        s_channel = images_hsv[:, 1:2, :, :]
        v_channel = images_hsv[:, 2:3, :, :]
        
        batch['img'] = torch.cat([h_channel, s_channel, v_channel], dim=1)
        
        return batch

    # Apply the patch to the correct method
    setattr(trainer_or_validator, method_name, types.MethodType(custom_preprocess, trainer_or_validator))

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
            task='classify',
            project=f'{weight.split("-")[0]}_{key}',
            name=weight.split('-')[1].split('.')[0],
            epochs=30, 
            imgsz=512, 
            device=0, 
            batch=32,
            # --- Disabling Augmentations ---
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