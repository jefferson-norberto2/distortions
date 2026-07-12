from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from distortions.utils.functions import generate_confusion_matrix
from tqdm import tqdm
import cv2

def run_yolo_accuracy_tests():
    yolo_names = ['26n', '26s', '26m', '26l', '26x']
    colors = {
            'RGB' : {'test': 'Datasets/LIST/test', 'cross': 'Datasets/CSIQ'}, 
            'LAB' : {'test': 'Datasets/LIST_LAB/test', 'cross': 'Datasets/CSIQ_LAB'},
            'HSV' : {'test': 'Datasets/LIST_HSV/test', 'cross': 'Datasets/CSIQ_HSV'}
            }


    for color, datasets in colors.items():
        for dataset_name, dataset_path in datasets.items():
            base_dir = f"runs/tested/yolo"
            for yolo_name in yolo_names:
                os.makedirs(base_dir, exist_ok=True)

                count = sum(1 for folder in os.listdir(base_dir) if folder.startswith('yolo'))
                        
                save_dir = f"{base_dir}/{yolo_name}/{color}/run_{count+1}"

                while os.path.exists(save_dir):
                    count += 1
                    save_dir = f"{base_dir}/{yolo_name}/{color}/run_{count+1}"

                os.makedirs(save_dir, exist_ok=True)

                # Carregar modelo
                model_path = f'runs/trained/yolo/{yolo_name}/{color}/best.pt'
                model = YOLO(model_path)

                # Definições
                classes = {0: 'awgn', 1: 'blur', 2: 'contrast', 3: 'fnoise', 4: 'jpeg', 5: 'jpeg2000', 6: 'src'}
                class_names = [classes[i] for i in range(len(classes))] 

                y_true = []
                y_pred = []

                print("Iniciando inferência...")
                img_path = ""

                for key, value in classes.items():
                    folder_path = os.path.join(dataset_path, value)
                    
                    if not os.path.exists(folder_path):
                        print(f"Pasta não encontrada: {folder_path}")
                        continue

                    images = os.listdir(folder_path)
                    total = len(images)
                    
                    for i, img_name in enumerate(tqdm(images, desc=f"Processing {value} images")):
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

                # --- Visualização com Scikit-Learn ---
                print("Gerando Matriz de Confusão...")

                generate_confusion_matrix(y_true, y_pred, class_names, 'runs/cm/yolo', f"{dataset_name.lower()}_{yolo_name.lower()}_{color.lower()}", None)
            
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

if __name__ == "__main__":
    run_yolo_accuracy_tests()