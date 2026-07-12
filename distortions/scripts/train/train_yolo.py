from ultralytics import YOLO
from distortions.utils.functions import extract_model_parts

def train_yolos():

    models = {
        'yolo26n': 'yolo26n-cls.pt',
        'yolo26s': 'yolo26s-cls.pt',
        'yolo26m': 'yolo26m-cls.pt',
        'yolo26l': 'yolo26l-cls.pt',
        'yolo26x': 'yolo26x-cls.pt'
    }

    colors_space = ['RGB','LAB', 'HSV']

    for color in colors_space:

        for model, weight in models.items():

            model = YOLO(weight)
            family, version = extract_model_parts(model)
            print(f"Training {family}{version} on color space {color}")

            model.train(
                data=f'Datasets/LIST_{color}', 
                task='classify',
                project=f'{family}/{version}/{color}',
                name='train',
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