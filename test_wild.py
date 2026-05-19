import os
import torch
import wandb
import yaml
import gc

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from distortions.dataset.single_dataset import SingleDataset
from distortions.model.custom_mobilenet import CustomMobileNet
from distortions.utils.functions import generate_confusion_matrix, check_predictions, save_results_local_and_wandb, extract_model_parts
from tqdm import tqdm

load_dotenv()

def make_save_dir(args):
    base = args['base_path']
    count = 1
    save_path = f"{base}/run_{count}"
    while os.path.exists(save_path):
        count += 1
        save_path = f"{base}/run_{count}"
        
    os.makedirs(save_path, exist_ok=True)
    return save_path


def test(args: dict):
    project_name = os.getenv("PROJECT_NAME")
    wandb.init(
        project=project_name,
        name=args['experiment_name'],
        config=args,
        reinit=True,
        mode=os.getenv("WANDB_MODE", "online")
    )

    try:
        save_path = make_save_dir(args)
        print(f"Results will be saved in: {save_path}")
        
        with open(args['args_path'], 'r') as file:
            data = yaml.safe_load(file)

        print(data['class_names'])
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_ds = SingleDataset(args['dataset_path'], image_mode=args['image_mode'], image_size=(args['imgsz'], args['imgsz']))
        
        test_ds.update_classes(data['class_names'])
        
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
        
        model = CustomMobileNet(num_classes=len(test_ds.class_names), pre_trained=False, backbone=args['model']).to(device)
        model.load_state_dict(torch.load(args['weights_path'], map_location=device, weights_only=True))
        model.eval()

        error_table = wandb.Table(columns=["Image", "True Class", "Predicted Class"])

        v_acc, v_total = 0, 0
        y_true, y_pred = [], []
                
        with torch.no_grad():
            pbar_val = tqdm(test_loader, desc=f"Testing {args['experiment_name']}: ")
            for x_rgb, labels in pbar_val:
                x_rgb, labels = x_rgb.to(device), labels.to(device)
                outputs = model(x_rgb)
                preds = outputs.argmax(1)
                                
                v_acc += (preds == labels).sum().item()
                v_total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        accuracy = v_acc / v_total * 100
        args['accuracy'] = accuracy
        print(f"Test Accuracy: {accuracy:.2f}%")

        check_predictions(y_true, y_pred, test_ds, save_path, error_table, wandb)
        generate_confusion_matrix(y_true, y_pred, test_ds.class_names, save_path, wandb)
        save_results_local_and_wandb(args, test_ds, y_true, y_pred, save_path, accuracy, wandb)

        wandb.log({"misclassified_samples": error_table})

    except Exception as e:
        print("An error occurred during testing:", str(e))
        return None 
    finally:
        wandb.finish()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    models = ['mobilenet_v1']
    colors = ['HSV']
    datasets = {
       # 'test': 'Datasets/LIST/test',
        # 'cross_test': '/media/jmn/Removable Disk/Datasets/CSIQ',
        'wild' : '/media/jmn/Removable Disk/Datasets/Wild'
    }

    # Dicionário para acumular leituras
    global_hardware_data = {
        m: {'ram': [], 'gpu': [], 'vram': [], 'power': []} for m in models
    }

    for folder_name, dataset_path in datasets.items():
        for color in colors:
            for model_name in models:
                family, version = extract_model_parts(model_name)

                hierarchical_path = f"runs/tested/{family}/{version}/{folder_name}/{color}"
                
                experiment_id = f"{version}_{folder_name}_{color}"
                
                args = {
                    "base_path": hierarchical_path,
                    "experiment_name": experiment_id,
                    "dataset_path" : dataset_path,
                    "imgsz": 400,
                    "weights_path": f'/media/jmn/Removable Disk/runs/trained/mobilenet/V1/HSV/best.pt', 
                    "model": model_name,
                    "image_mode": color,
                    "evaluation_folder": folder_name,
                    "args_path": f'/media/jmn/Removable Disk/runs/trained/mobilenet/V1/HSV/args.yaml',
                }

                test(args)