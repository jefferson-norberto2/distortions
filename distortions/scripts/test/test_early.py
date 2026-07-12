import os
import torch
import wandb
import yaml
import gc

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from distortions.dataset.dual_dataset import DualDataset

# 1. Update the import to load the Early Fusion model
from distortions.model.early_fusion import EarlyFusionAdapter
from distortions.utils.functions import generate_confusion_matrix, check_predictions, save_results_local_and_wandb
from distortions.utils.hardware import HardwareProfiler
from tqdm import tqdm

load_dotenv()

# 2. Update family identifier
FAMILY, VERSION = "early_fusion", "v1"

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

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_ds = DualDataset(args['dataset_path'], image_mode=args['image_mode'], image_size=(args['imgsz'], args['imgsz']))
        
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
        
        # 3. Instantiate the EarlyFusionAdapter using the YAML factory method
        model = EarlyFusionAdapter(num_classes=7).to(device)
        model.load_state_dict(torch.load(args['weights_path'], map_location=device, weights_only=True))
        model.eval()

        error_table = wandb.Table(columns=["Image", "True Class", "Predicted Class"])

        v_acc, v_total = 0, 0
        y_true, y_pred = [], []
        
        profiler = HardwareProfiler(device_index=0)

        print("Running GPU warm-up...")
        # The Early Fusion model still expects two inputs (RGB and HSV) 
        # because the concatenation happens inside the forward pass.
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        for _ in range(10):
            _ = model(dummy_input, dummy_input)
        
        with torch.no_grad():
            pbar_val = tqdm(test_loader, desc=f"Testing {args['experiment_name']}: ")
            for x_rgb, x_hsv, labels in pbar_val:
                x_rgb, x_hsv, labels = x_rgb.to(device), x_hsv.to(device), labels.to(device)
                outputs = model(x_rgb, x_hsv)
                preds = outputs.argmax(1)
                
                profiler.sample()
                
                v_acc += (preds == labels).sum().item()
                v_total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        accuracy = v_acc / v_total * 100
        args['accuracy'] = accuracy
        print(f"Test Accuracy: {accuracy:.2f}%")

        # Keep saving locally (optional, but highly recommended)
        hw_metrics = profiler.save_to_yaml(save_path)
        wandb.log({"hardware_averages": hw_metrics})

        check_predictions(y_true, y_pred, test_ds, save_path, error_table, wandb)
        generate_confusion_matrix(y_true, y_pred, test_ds.class_names, save_path, wandb)
        save_results_local_and_wandb(args, test_ds, y_true, y_pred, save_path, accuracy, wandb)

        wandb.log({"misclassified_samples": error_table})

        # RETURN RAW DATA TO THE MAIN BLOCK
        return profiler.get_raw_data()

    except Exception as e:
        print("An error occurred during testing:", str(e))
        return None 
    finally:
        wandb.finish()
        del model 
        gc.collect()
        torch.cuda.empty_cache()

def run_early_fusion_tests():
    # 4. Updated the model list to loop over the early fusion model
    models = ['early_fusion']
    colors = ['HSV']
    datasets = {
        'test': '/media/jmn/Removable Disk/Datasets/LIST/test',
        'cross_test': '/media/jmn/Removable Disk/Datasets/CSIQ',
    }

    # Dictionary to accumulate hardware readings globally
    global_hardware_data = {
        m: {'ram': [], 'gpu': [], 'vram': [], 'power': []} for m in models
    }

    for folder_name, dataset_path in datasets.items():
        for color in colors:
            for model_name in models:

                hierarchical_path = f"runs/tested/{FAMILY}/{VERSION}/{folder_name}/{color}"
                experiment_id = f"{VERSION}_{folder_name}_{color}"
                
                # 5. Make sure these paths point to your newly trained Early Fusion weights and args!
                # I updated the path strings to reflect the 'early_fusion' folder structure.
                args = {
                    "imgsz": 512,
                    "base_path": hierarchical_path,
                    "experiment_name": experiment_id,
                    "dataset_path" : dataset_path,
                    "weights_path": "runs/trained/early_fusion/v1/train1/best.pt", 
                    "model": model_name,
                    "image_mode": color,
                    "evaluation_folder": folder_name,
                    "args_path": "runs/trained/early_fusion/v1/train1/args.yaml"
                }

                raw_data = test(args)
                
                if raw_data is not None:
                    global_hardware_data[model_name]['ram'].extend(raw_data['ram'])
                    global_hardware_data[model_name]['gpu'].extend(raw_data['gpu'])
                    global_hardware_data[model_name]['vram'].extend(raw_data['vram'])
                    global_hardware_data[model_name]['power'].extend(raw_data['power'])

    # --- FINAL GLOBAL PROCESSING PER MODEL ---
    print("\n--- Generating Global Hardware Reports ---")
    avg = lambda x: sum(x) / len(x) if x else 0.0

    for model_name, data in global_hardware_data.items():
        if data['power']: 
            summary = {
                "System_RAM_Usage_GB": {
                    "average": round(avg(data['ram']), 2),
                    "peak": round(max(data['ram']), 2)
                },
                "GPU_Processing_Usage_Percent": {
                    "average": round(avg(data['gpu']), 2),
                    "peak": round(max(data['gpu']), 2)
                },
                "GPU_VRAM_Allocation_GB": {
                    "average": round(avg(data['vram']), 2),
                    "peak": round(max(data['vram']), 2)
                },
                "Power_Consumption_Watts": {
                    "average": round(avg(data['power']), 2),
                    "peak": round(max(data['power']), 2)
                }
            }

            model_dir = f"runs/tested/{FAMILY}/{VERSION}"
            os.makedirs(model_dir, exist_ok=True)
            
            summary_path = f"{model_dir}/global_hardware_metrics.yaml"
            
            # Saving the file
            with open(summary_path, 'w') as f:
                yaml.dump({model_name: summary}, f, default_flow_style=False, sort_keys=False)
                
            print(f"[{VERSION}] Global hardware summary saved to: {summary_path}")

if __name__ == "__main__":
    run_early_fusion_tests()