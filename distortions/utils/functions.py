import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import yaml
from collections import defaultdict

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image as PILImage

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def generate_confusion_matrix(y_true, y_pred, class_names, save_path, wandb=None):
    def remove_zero_texts(disp):
        for text_obj in disp.text_.ravel():
            if text_obj.get_text() in ['0', '0.0', '0.00']:
                text_obj.set_text('')

    cm = confusion_matrix(y_true, y_pred)
    cm_transposed = cm.T 
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_transposed, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    remove_zero_texts(disp)
    
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')

    standard_cm_path = f"{save_path}/confusion_matrix.png"
    plt.savefig(standard_cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    cm_normalized_transposed = np.round(cm_normalized.T, 2)
    
    fig_norm, ax_norm = plt.subplots(figsize=(10, 8))
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized_transposed, display_labels=class_names)
    disp_normalized.plot(cmap=plt.cm.Blues, ax=ax_norm, xticks_rotation='vertical')
    remove_zero_texts(disp_normalized)

    ax_norm.set_xlabel('True')
    ax_norm.set_ylabel('Predicted')

    normalized_cm_path = f"{save_path}/confusion_matrix_normalized.png"
    plt.savefig(normalized_cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig_norm)

    if wandb is not None:
        wandb.log({
            "confusion_matrix/absolute": wandb.Image(standard_cm_path),
            "confusion_matrix/normalized": wandb.Image(normalized_cm_path)
        })

def check_predictions(y_true, y_pred, test_ds, save_path, error_table=None, wandb=None, max_logs_per_class=10):
    
    wandb_log_counts = {}

    for i in range(len(y_true)):
        pred_class = test_ds.class_names[y_pred[i]]
        true_class = test_ds.class_names[y_true[i]]
        
        if pred_class != true_class:
            error_dir = f"{save_path}/errors/{true_class}_as_{pred_class}"
            os.makedirs(error_dir, exist_ok=True)
                
            img_path = test_ds.samples[i][0]
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, f"{error_dir}/{img_name}")

            if wandb is not None and error_table is not None:
                if true_class not in wandb_log_counts:
                    wandb_log_counts[true_class] = 0
                
                if wandb_log_counts[true_class] < max_logs_per_class:
                    img = PILImage.open(img_path)
                    img.thumbnail((256, 256)) 
                    
                    error_table.add_data(
                        wandb.Image(img), 
                        true_class, 
                        pred_class
                    )
                    
                    wandb_log_counts[true_class] += 1

def save_results_local_and_wandb(args, test_ds, y_true, y_pred, save_path, accuracy, wandb=None):
    correct_total = 0
    incorrect_total = 0
    samples_per_class = defaultdict(int)
    correct_per_class = defaultdict(int)

    for true_idx, pred_idx in zip(y_true, y_pred):
        true_class_name = test_ds.class_names[true_idx]
        
        samples_per_class[true_class_name] += 1
        
        if true_idx == pred_idx:
            correct_total += 1
            correct_per_class[true_class_name] += 1
        else:
            incorrect_total += 1

    samples_per_class = dict(samples_per_class)
    correct_per_class = dict(correct_per_class)

    if wandb is not None:
        wandb.log({
            "metrics/accuracy": accuracy,
            "metrics/total_correct": correct_total,
            "metrics/total_incorrect": incorrect_total,
        })

    resolution_str = "Unknown"
    if len(test_ds.samples) > 0:
        img_path = test_ds.samples[0][0]
        with PILImage.open(img_path) as img:
            wi, hi = img.size
            resolution_str = f"{wi}x{hi}"
        
        if wandb is not None:
            wandb.config.update({"resolution": resolution_str})

    info_dict = {
        "Accuracy_percent": float(f"{accuracy:.2f}"),
        "Correct": correct_total,
        "Incorrect": incorrect_total,
        "Samples": len(y_true),
        "Samples_per_Class": samples_per_class,
        "Correct_per_Class": correct_per_class,
        "Dataset": args.get('dataset_path', 'Unknown'),
        "Classes": test_ds.class_names,
        "Model": args.get('model', 'Unknown'),
        "Weights": args.get('weights_path', 'Unknown'),
        "Resolution": resolution_str
    }

    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/informations.yaml", "w") as f:
        yaml.dump(info_dict, f, default_flow_style=False, sort_keys=False)

def extract_model_parts(model_name: str):
    if 'mobilenet' in model_name.lower():
        family = 'mobilenet'
        parts = model_name.split('_')
        version = ''
        for i in range(1, len(parts)):
            version += parts[i]
            version += '_' if parts[i] != parts[-1] else ''            
    elif 'yolo' in model_name.lower():
        family = 'yolo'
        parts = model_name.split(family, 1)
        version = parts[1]
    elif 'resnet' in model_name.lower():
        family = 'resnet'
        parts = model_name.split(family, 1)
        version = parts[1]
    else:
        raise ValueError(f"Model name '{model_name}' does not match expected patterns for MobileNet, YOLO, or ResNet.")
        
    return family, version