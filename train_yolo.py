from ultralytics import YOLO

weights = [
    'yolo26n-cls.pt',
    'yolo26s-cls.pt',
    'yolo26m-cls.pt',
    'yolo26l-cls.pt',
    'yolo26x-cls.pt'
]

for weight in weights:

    model = YOLO(weight)  

    model.train(
        data='Datasets/LIST', 
        task='classify',
        project=weight.split('-')[0],  # Usar o nome do modelo como subpasta
        name=weight.split('-')[1].split('.')[0],
        epochs=30, 
        imgsz=512, 
        device=0, 
        batch=32,
        # --- Desativando Augmentations ---
        hsv_h=0.0,      # Ajuste de matiz (hue)
        hsv_s=0.0,      # Ajuste de saturação
        hsv_v=0.0,      # Ajuste de valor (brilho)
        degrees=0.0,    # Rotação
        translate=0.0,  # Translação
        scale=0.0,      # Escala (ganho/perda de zoom)
        shear=0.0,      # Cisalhamento
        perspective=0.0,# Perspectiva
        flipud=0.0,     # Inversão vertical
        fliplr=0.5,     # Inversão horizontal 
        mosaic=0.0,     # Desativa o mosaico (importante para detecção)
        mixup=0.0,      # Desativa o MixUp (comum em classificação)
        auto_augment=None, # Desativa políticas automáticas (como RandAugment)
        erasing=0.0     # Desativa o Random Erasing
    )