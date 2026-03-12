from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from distortions.utils.bar_progress import BarProgress

base_dir = "runs/test"
os.makedirs(base_dir, exist_ok=True)

count = sum(1 for folder in os.listdir(base_dir) if folder.startswith('yolo'))
        
save_dir = f"{base_dir}/yolo_{count+1}"
os.makedirs(save_dir, exist_ok=True)

# Carregar modelo
model = YOLO('runs/classify/train5/weights/best.pt')

# Definições
classes = {0: 'awgn', 1: 'blur', 2: 'contrast', 3: 'fnoise', 4: 'jpeg', 5: 'jpeg2000', 6: 'src'}
class_names = [classes[i] for i in range(len(classes))] # [0, 1, 2, 3] -> nomes
base_path = 'Datasets/CSIQ/train'

y_true = []
y_pred = []

print("Iniciando inferência...")

bar = BarProgress(header=['Current step', 'Total steps'], width=12)

for key, value in classes.items():
    folder_path = os.path.join(base_path, value)
    
    if not os.path.exists(folder_path):
        print(f"Pasta não encontrada: {folder_path}")
        continue

    images = os.listdir(folder_path)
    total = len(images)
    task_id = bar.start(total)
    
    for i, img_name in enumerate(images):
        # Filtro básico de extensão
        if not img_name.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(folder_path, img_name)
    
        results = model.predict(
            source=img_path,
            device=0,
            imgsz=512, 
            save=False,
            verbose=False,
        )

        result = results[0]
        
        y_true.append(key)              # Classe real (baseada na pasta)
        y_pred.append(result.probs.top1) # Classe predita (baseada no modelo)

        if key != result.probs.top1:
            # Salvar a imagem com erro para análise posterior
            error_dir = f"{save_dir}/errors/{classes[key]}_as_{classes[result.probs.top1]}"
            os.makedirs(error_dir, exist_ok=True)
            plt.imsave(os.path.join(error_dir, img_name), plt.imread(img_path)) 

        bar.update(
            task_id=task_id, 
            advance=1, 
            description=[i+1, total]
        )
    bar.stop()

# --- Visualização com Scikit-Learn ---
print("Gerando Matriz de Confusão...")

cm = confusion_matrix(y_pred, y_true)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)

# Este é o bloco mágico que remove os zeros
for text_obj in disp.text_.ravel():
    if text_obj.get_text() == '0':
        text_obj.set_text('') # Deixa a célula sem texto algum

plt.xlabel('True Label')     
plt.ylabel('Predicted Label') 
plt.savefig(f"{save_dir}/confusion_matrix_yolo.png", dpi=300)

# calculo de acurácia
accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true) * 100
print(f"Acurácia: {accuracy:.2f}%")
with open(f"{save_dir}/accuracy.txt", "w") as f:
    f.write(f"Acurácia: {accuracy:.2f}%\n")