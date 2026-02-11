from ultralytics import YOLO

def augment(image):
    return image

model = YOLO('yolo26m-cls.pt') 

results = model.train(
    data='/home/jmn/dev/Datasets/ECSIQ_512/', 
    epochs=20, 
    imgsz=720, 
    device=0, 
    batch=8,
    # --- Desativando Augmentations ---
    hsv_h=0.0,      # Ajuste de matiz (hue)
    hsv_s=0.0,      # Ajuste de saturação
    hsv_v=0.0,      # Ajuste de valor (brilho)
    degrees=0.0,    # Rotação
    translate=0.0,  # Translação
    scale=0.0,      # Escala (ganho/perda de zoom)
    shear=0.0,      # Cisalhamento
    perspective=0.0,# Perspectiva
    flipud=0.0,     # Inversão vertical (vire de cabeça para baixo)
    fliplr=0.0,     # Inversão horizontal (espelhamento)
    mosaic=0.0,     # Desativa o mosaico (importante para detecção, mas bom zerar)
    mixup=0.0,      # Desativa o MixUp (comum em classificação)
    auto_augment=None, # Desativa políticas automáticas (como RandAugment)
    erasing=0.0     # Desativa o Random Erasing
)