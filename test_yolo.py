from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO('runs/classify/train4/weights/best.pt')

# Valida o modelo na base de dados LIVE
results = model.val(
    data='/home/jmn/dev/python/distortions/data/LIVE_croped',
    imgsz=512,
    batch=64,
    device=0
)

# Exibe os resultados
#print(results)