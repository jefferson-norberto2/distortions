import os
import torch
import cv2
import shutil
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from distortions.dataset.single_dataset import SingleDataset
from distortions.model.custom_mobilenet import CustomMobileNetV3
from tqdm import tqdm

def generate_confusion_matrix(y_true, y_pred, class_names, save_path):
    def remove_zero_texts(disp):
        for text_obj in disp.text_.ravel():
            if text_obj.get_text() == '0':
                text_obj.set_text('')

    cm = confusion_matrix(y_pred, y_true)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)

    remove_zero_texts(disp)

    plt.xlabel('True')     
    plt.ylabel('Predicted') 
    plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300)
    plt.close()

    cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    # make 2 decimal places
    cm_normalized = np.round(cm_normalized, 2)
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
    disp_normalized.plot(cmap=plt.cm.Blues)

    remove_zero_texts(disp_normalized)

    plt.xlabel('True')     
    plt.ylabel('Predicted') 
    plt.savefig(f"{save_path}/confusion_matrix_normalized.png", dpi=300)
    plt.close()

def check_predictions(y_true, y_pred, test_ds, save_path):
    for i in range(len(y_true)):
        pred_class = test_ds.class_names[y_pred[i]]
        true_class = test_ds.class_names[y_true[i]]
        if pred_class != true_class:
            error_dir = f"{save_path}/errors/{true_class}_as_{pred_class}"
            os.makedirs(error_dir, exist_ok=True)
                
            img_path = test_ds.samples[i][0]
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, f"{error_dir}/{img_name}")

def make_save_dir(args):
    save_path = f"{args['base_path']}/{args['experiment_name']}"
    count = 1
    while os.path.exists(save_path):
        save_path = f"{args['base_path']}/{args['experiment_name']}_{count}"
        count += 1
    os.makedirs(save_path, exist_ok=True)
    return save_path

def save_results(args, test_ds, y_true, y_pred, save_path):
    with open(f"{save_path}/informations.yaml", "w") as f:
        f.write(f"Accuracy: {args['accuracy']:.2f}%\n")
        f.write(f"Correct: {sum(1 for true, pred in zip(y_true, y_pred) if true == pred)}\n")
        f.write(f"Incorrect: {sum(1 for true, pred in zip(y_true, y_pred) if true != pred)}\n")
        f.write(f"Samples: {len(y_true)}\n")
        f.write(f"Samples per Class: { {test_ds.class_names[i]: y_true.count(i) for i in range(len(test_ds.class_names))} }\n")
        f.write(f"Correct per Class: { {test_ds.class_names[i]: sum(1 for true, pred in zip(y_true, y_pred) if true == pred == i) for i in range(len(test_ds.class_names))} }\n")
        f.write(f"Incorrect per Class: { {test_ds.class_names[i]: sum(1 for true, pred in zip(y_true, y_pred) if true == i and pred != i) for i in range(len(test_ds.class_names))} }\n")
        f.write(f"Dataset: {args['dataset_path']}\n")
        f.write(f"Classes: {test_ds.class_names}\n")
        f.write(f"Model: {args['model']}\n")
        f.write(f"Weights: {args['weights_path']}\n")

        if len(test_ds.samples) > 0:
            img_path = test_ds.samples[0][0]
            img_cv = cv2.imread(img_path)
            hi, wi, _ = img_cv.shape
            f.write(f"Resolution: {wi}x{hi}\n")

def test(args: dict):
    try:
        save_path = make_save_dir(args)
        print(f"Results will be saved in: {save_path}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_ds = SingleDataset(args['dataset_path'], image_mode=args['image_mode'])
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
        model = CustomMobileNetV3(num_classes=len(test_ds.class_names), pre_trained=False, backbone=args['model']).to(device)
        model.load_state_dict(torch.load(args['weights_path'], map_location=device, weights_only=True))
        model.eval()

        v_acc, v_total = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            pbar_val = tqdm(test_loader, desc="Testing: ")
            for x_rgb, labels in pbar_val:
                x_rgb, labels = x_rgb.to(device), labels.to(device)
                outputs = model(x_rgb)
                preds = outputs.argmax(1)
                v_acc += (preds == labels).sum().item()
                v_total += labels.size(0)
                y_true.extend(labels.cpu().numpy()); y_pred.extend(preds.cpu().numpy())

        check_predictions(y_true, y_pred, test_ds, save_path)
            
        accuracy = v_acc / v_total * 100
        args['accuracy'] = accuracy

        print(f"Test Accuracy: {accuracy:.2f}%")

        generate_confusion_matrix(y_true, y_pred, test_ds.class_names, save_path)
        save_results(args, test_ds, y_true, y_pred, save_path)
    except Exception as e:
        print("An error occurred during testing:", str(e))


if __name__ == "__main__":
        models = ['mobilenet_v3_small', 'mobilenet_v3_large']
        colors = ['LAB', 'HSV']
        datasets = {
            'test': 'Datasets/LIST/test',
            'cross_test': 'Datasets/CSIQ',
        }

        for folder_name, dataset_path in datasets.items():
            for color in colors:
                for model_name in models:
                    args = {
                        "base_path": f'runs/{folder_name}',        
                        "experiment_name": f"{model_name}_{color}",
                        "dataset_path" : dataset_path,
                        "weights_path": f'runs/{model_name}_{color}/train1/best.pt',
                        "model": model_name,
                        "image_mode": color,
                    }

                test(args)
        