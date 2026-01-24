from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO('runs/classify/train14/weights/best.pt')

# Valida o modelo na base de dados LIVE
results = model.val(
    data='/root/Documents/dev/python/distortions/data/LIVE',
    imgsz=256,
    batch=64,
    device=0
)

# Exibe os resultados
#print(results)