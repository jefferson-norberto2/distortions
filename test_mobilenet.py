import os
import torch
import wandb

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from distortions.dataset.single_dataset import SingleDataset
from distortions.model.custom_mobilenet import CustomMobileNet
from distortions.utils.functions import generate_confusion_matrix, check_predictions, save_results_local_and_wandb
from tqdm import tqdm

load_dotenv()

def make_save_dir(args):
    save_path = f"{args['base_path']}/{args['experiment_name']}"
    count = 1
    while os.path.exists(save_path):
        save_path = f"{args['base_path']}/{args['experiment_name']}_{count}"
        count += 1
    os.makedirs(save_path, exist_ok=True)
    return save_path

def test(args: dict):
    project_name = os.getenv("PROJECT_NAME")
    wandb.init(
        project=project_name,
        name=args['experiment_name'],
        config=args,
        reinit=True
    )

    try:
        save_path = make_save_dir(args)
        print(f"Results will be saved in: {save_path}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_ds = SingleDataset(args['dataset_path'], image_mode=args['image_mode'])
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

        # Process errors, matrices and metrics
        check_predictions(y_true, y_pred, test_ds, save_path, error_table, wandb)
        generate_confusion_matrix(y_true, y_pred, test_ds.class_names, save_path, wandb)
        save_results_local_and_wandb(args, test_ds, y_true, y_pred, save_path, accuracy, wandb)

        wandb.log({"misclassified_samples": error_table})

    except Exception as e:
        print("An error occurred during testing:", str(e))
    finally:
        wandb.finish()


if __name__ == "__main__":
    models = ['mobilenet_v2']
    colors = ['RGB', 'LAB', 'HSV']
    datasets = {
        'test': 'Datasets/LIST/test',
        'cross_test': 'Datasets/CSIQ',
    }

    for folder_name, dataset_path in datasets.items():
        for color in colors:
            for model_name in models:
                experiment_id = f"{folder_name}_{model_name}_{color}"
                
                args = {
                    "base_path": f'runs/{folder_name}',
                    "experiment_name": experiment_id,
                    "dataset_path" : dataset_path,
                    "weights_path": f'runs/{model_name}_{color}/train1/best.pt',
                    "model": model_name,
                    "image_mode": color,
                    "evaluation_folder": folder_name
                }

                test(args)