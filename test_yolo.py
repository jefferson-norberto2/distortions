from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from distortions.utils.progress import BarProgress
import cv2
import numpy as np

yolo_names = ['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x']
colors = ['LAB', 'HSV']
datasets = {
    'test': 'Datasets/LIST/test',
    'cross_test': 'Datasets/CSIQ',
}

for dataset_name, dataset_path in datasets.items():
    base_dir = f"runs/{dataset_name}"
    for color in colors:
        for yolo_name in yolo_names:
            os.makedirs(base_dir, exist_ok=True)

            count = sum(1 for folder in os.listdir(base_dir) if folder.startswith('yolo'))
                    
            save_dir = f"{base_dir}/{yolo_name}_{color}"

            while os.path.exists(save_dir):
                count += 1
                save_dir = f"{base_dir}/{yolo_name}_{color}_{count+1}"

            os.makedirs(save_dir, exist_ok=True)

            # Carregar modelo
            model_path = f'runs/classify/{yolo_name}_{color}/cls/weights/best.pt'
            model = YOLO(model_path)

            # Definições
            classes = {0: 'awgn', 1: 'blur', 2: 'contrast', 3: 'fnoise', 4: 'jpeg', 5: 'jpeg2000', 6: 'src'}
            class_names = [classes[i] for i in range(len(classes))] 

            y_true = []
            y_pred = []

            print("Iniciando inferência...")

            bar = BarProgress(header=['Current step', 'Total steps'], width=12)

            img_path = ""

            for key, value in classes.items():
                folder_path = os.path.join(dataset_path, value)
                
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

                    img_bgr = cv2.imread(img_path)
                    
                    if img_bgr is None:
                        continue

                    if color == 'HSV':
                        img_target = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                    elif color == 'LAB':
                        img_target = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                    else:
                        img_target = img_bgr
                
                    results = model.predict(
                        source=img_target,
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

            for text_obj in disp.text_.ravel():
                if text_obj.get_text() == '0':
                    text_obj.set_text('')
            

            plt.xlabel('True')     
            plt.ylabel('Predicted') 
            plt.savefig(f"{save_dir}/confusion_matrix_yolo.png", dpi=300)
            
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
            disp_normalized.plot(cmap=plt.cm.Blues)

            for text_obj in disp_normalized.text_.ravel():
                if text_obj.get_text() == '0.00':
                    text_obj.set_text('')
            
            plt.xlabel('True')     
            plt.ylabel('Predicted') 
            plt.savefig(f"{save_dir}/confusion_matrix_yolo_normalized.png", dpi=300)
            plt.close()

            img = cv2.imread(img_path)
            hi, wi, _ = img.shape

            # calculo de acurácia
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
                f.write(f"Dataset base: {dataset_path}\n")
                f.write(f"Resolution: {wi}x{hi}\n")
                f.write(f"Classes: {class_names}\n")
                f.write(f"Modelo: {model_path}\n")
                f.write(f"Modelo base: {yolo_name}\n")
                f.write(f"Espaço de cor: {color}\n")

            print("Processo concluído! Resultados salvos em:", save_dir)


