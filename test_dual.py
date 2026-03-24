import os
import torch
import cv2
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from distortions.utils.progress import BarProgress
from train_dual import YOLO26DualStream

# IMPORTANTE: Você precisa importar a classe do seu modelo aqui
# from seu_arquivo_de_treino import YOLO26DualStreamAttention

# ==========================================
# 1. CONFIGURAÇÕES INICIAIS
# ==========================================
base_dir = "runs/test"
os.makedirs(base_dir, exist_ok=True)

count = sum(1 for folder in os.listdir(base_dir) if folder.startswith('dualstream'))
save_dir = f"{base_dir}/dualstream_{count+1}"

while os.path.exists(save_dir):
    count += 1
    save_dir = f"{base_dir}/dualstream_{count+1}"

os.makedirs(save_dir, exist_ok=True)

# Definições de Classes e Caminhos
base_path = 'Datasets/CIST/test' # Ajuste para a sua base de teste
classes = {0: 'awgn', 1: 'blur', 2: 'contrast', 3: 'fnoise', 4: 'jpeg', 5: 'jpeg2000', 6: 'src'}
class_names = [classes[i] for i in range(len(classes))]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. CARREGAMENTO DO MODELO E TRANSFORMS
# ==========================================
print("Carregando modelo Dual-Stream...")
model_path = 'runs/dual_stream/train6/best.pt' # Aponte para o seu melhor peso

# Instancia a classe (certifique-se de que ela está acessível neste script)
# Assumindo que num_classes é o tamanho do seu dicionário
model = YOLO26DualStream('yolo26m-cls.pt', num_classes=len(classes), imgsz=512)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval() # Modo de inferência estrito (desliga Dropout, fixa BatchNorm)

# Transformações idênticas às usadas no treinamento
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. LOOP DE INFERÊNCIA
# ==========================================
y_true = []
y_pred = []

print("Iniciando inferência...")
bar = BarProgress(header=['Current step', 'Total steps'], width=12)

img_path_for_info = ""

# Usar autocast na inferência também acelera o teste na RTX 4050
with torch.no_grad(), torch.amp.autocast('cuda'):
    for key, value in classes.items():
        folder_path = os.path.join(base_path, value)
        
        if not os.path.exists(folder_path):
            print(f"Pasta não encontrada: {folder_path}")
            continue

        images = os.listdir(folder_path)
        total = len(images)
        task_id = bar.start(total)
        
        for i, img_name in enumerate(images):
            if not img_name.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(folder_path, img_name)
            img_path_for_info = img_path # Guarda a última para pegar a resolução no final
            
            # 1. Carrega e converte a imagem
            img_rgb_pil = Image.open(img_path).convert('RGB')
            img_hsv_pil = img_rgb_pil.convert('HSV')
            
            # 2. Aplica transforms e adiciona a dimensão do batch (1, C, H, W)
            t_rgb = transform(img_rgb_pil).unsqueeze(0).to(device)
            t_hsv = transform(img_hsv_pil).unsqueeze(0).to(device)
            
            # 3. Inferência
            outputs = model(t_rgb, t_hsv)
            pred_class = outputs.argmax(1).item() # Pega o índice com maior probabilidade
            
            y_true.append(key)
            y_pred.append(pred_class)

            # 4. Salvar erros com shutil.copy (Preserva os artefatos originais da distorção)
            if key != pred_class:
                error_dir = f"{save_dir}/errors/{classes[key]}_as_{classes[pred_class]}"
                os.makedirs(error_dir, exist_ok=True)
                shutil.copy(img_path, os.path.join(error_dir, img_name))

            bar.update(task_id=task_id, advance=1, description=[i+1, total])
        bar.stop()

# ==========================================
# 4. MÉTRICAS E EXPORTAÇÃO (Seu código original mantido)
# ==========================================
print("Gerando Matriz de Confusão...")

cm = confusion_matrix(y_true, y_pred) # Corrigido: y_true primeiro (padrão Scikit)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)

for text_obj in disp.text_.ravel():
    if text_obj.get_text() == '0':
        text_obj.set_text('') 

plt.xlabel('Predicted Label')     
plt.ylabel('True Label') 
plt.savefig(f"{save_dir}/confusion_matrix_dualstream.png", dpi=300)
plt.close()

# Pegar resolução da última imagem processada
if img_path_for_info:
    img_cv = cv2.imread(img_path_for_info)
    hi, wi, _ = img_cv.shape
else:
    hi, wi = 0, 0

# Cálculo de acurácia
accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true) * 100
print(f"Acurácia: {accuracy:.2f}%")

with open(f"{save_dir}/informations.yaml", "w") as f:
    f.write(f"Acurácia: {accuracy:.2f}%\n")
    f.write(f"Acertos: {sum(1 for true, pred in zip(y_true, y_pred) if true == pred)}\n")
    f.write(f"Erros: {sum(1 for true, pred in zip(y_true, y_pred) if true != pred)}\n")
    f.write(f"Total de amostras: {len(y_true)}\n")
    f.write(f"Amostras por classe: { {class_names[i]: y_true.count(i) for i in range(len(class_names))} }\n")
    f.write(f"Acertos por classe: { {class_names[i]: sum(1 for true, pred in zip(y_true, y_pred) if true == pred == i) for i in range(len(class_names))} }\n")
    f.write(f"Erros por classe: { {class_names[i]: sum(1 for true, pred in zip(y_true, y_pred) if true == i and pred != i) for i in range(len(class_names))} }\n")
    f.write(f"Dataset base: {base_path}\n")
    f.write(f"Resolution: {wi}x{hi}\n")
    f.write(f"Classes: {class_names}\n")
    f.write(f"Modelo: {model_path}\n")

print("Processo concluído! Resultados salvos em:", save_dir)