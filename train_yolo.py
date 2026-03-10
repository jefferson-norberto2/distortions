from ultralytics import YOLO

def augment(image):
    return image

model = YOLO('yolo26m-cls.pt') 

results = model.train(
    data='/home/jmn/Dev/Datasets/Distortions_v3/', 
    epochs=5, 
    imgsz=480, 
    device=0, 
    batch=24,
    save=True,        
    save_period=1,    
    # --- Desativando Augmentations ---
    hsv_h=0.0,      # Ajuste de matiz (hue)
    hsv_s=0.0,      # Ajuste de saturação
    hsv_v=0.0,      # Ajuste de valor (brilho)
    degrees=0.0,    # Rotação
    translate=0.1,  # Translação
    scale=0.2,      # Escala (ganho/perda de zoom)
    shear=0.0,      # Cisalhamento
    perspective=0.0,# Perspectiva
    flipud=0.0,     # Inversão vertical
    fliplr=0.0,     # Inversão horizontal 
    mosaic=0.0,     # Desativa o mosaico (importante para detecção)
    mixup=0.1,      # Desativa o MixUp (comum em classificação)
    auto_augment=None, # Desativa políticas automáticas (como RandAugment)
    erasing=0.2     # Desativa o Random Erasing
)