from ultralytics import YOLO

def augment(image):
    return image

model = YOLO('yolo26m-cls.pt') 

model.train(
    data='Datasets/Dist_HSV', 
    epochs=30, 
    imgsz=512, 
    device=0, 
    batch=16,
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

model2 = YOLO('yolo26m-cls.pt') 

model2.train(
    data='Datasets/Dist_LAB', 
    epochs=30, 
    imgsz=512, 
    device=0, 
    batch=16,
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