from ultralytics import YOLO

weights = [
    'yolo26n-cls.pt',
    'yolo26s-cls.pt',
    'yolo26m-cls.pt',
    'yolo26l-cls.pt',
    'yolo26x-cls.pt'
]

colors_space = ['LAB', 'HSV']

for color in colors_space:

    for weight in weights:

        model = YOLO(weight)  

        model.train(
            data=f'Datasets/LIST_{color}', 
            task='classify',
            project=f'{weight.split("-")[0]}_{color}',
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