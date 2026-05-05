import os
import torch
import wandb
import yaml
import gc

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from distortions.dataset.yolo_dataset import YOLODataset
from ultralytics import YOLO
from distortions.utils.functions import generate_confusion_matrix, check_predictions, save_results_local_and_wandb, extract_model_parts
from distortions.utils.hardware import HardwareProfiler
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
    project_name = os.getenv("PROJECT_NAME", "Distortions_Evaluation")
    wandb.init(
        project=project_name,
        name=args['experiment_name'],
        config=args,
        reinit=True,
        mode=os.getenv("WANDB_MODE", "online")
    )

    try:
        if not args['hardware_evaluation']:
            if args['image_mode'] == 'LAB':
                args['dataset_path'] = args['dataset_path'].replace('LIST', 'LIST_LAB').replace('CSIQ', 'CSIQ_LAB')
            elif args['image_mode'] == 'HSV':
                args['dataset_path'] = args['dataset_path'].replace('LIST', 'LIST_HSV').replace('CSIQ', 'CSIQ_HSV')

        save_path = make_save_dir(args)
        print(f"Results will be saved in: {save_path}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_ds = YOLODataset(args['dataset_path'], image_mode=args['image_mode'], hardware_evaluation=args['hardware_evaluation'])
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

        
        model = YOLO(args['weights_path'])
        model.to(device)

        error_table = wandb.Table(columns=["Image", "True Class", "Predicted Class"])

        v_acc, v_total = 0, 0
        y_true, y_pred = [], []
        
        profiler = HardwareProfiler(device_index=0)

        print("Running GPU warm-up...")
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        dummy_input_normalized = dummy_input / 255.0
        for _ in range(10):
            _ = model(dummy_input_normalized, verbose=False)
        
        with torch.no_grad():
            pbar_val = tqdm(test_loader, desc=f"Testing {args['experiment_name']}: ")
            for x_rgb, labels in pbar_val:
                labels = labels.to(device)
                
                # Check if x_rgb is a tensor before moving to device
                # If it's a tuple of paths, YOLO can handle it directly
                if isinstance(x_rgb, torch.Tensor):
                    x_rgb = x_rgb.to(device)
                    
                # Coleta hardware a cada imagem processada
                profiler.sample()
                
                outputs = model(x_rgb, verbose=False) 
                
                preds_list = [r.probs.top1 for r in outputs]
                preds = torch.tensor(preds_list, device=device)
                
                v_acc += (preds == labels).sum().item()
                v_total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        accuracy = v_acc / v_total * 100
        args['accuracy'] = accuracy
        print(f"Test Accuracy: {accuracy:.2f}%")

        # Mantém salvando o local (opcional, mas bom ter)
        hw_metrics = profiler.save_to_yaml(save_path)
        wandb.log({"hardware_averages": hw_metrics})

        check_predictions(y_true, y_pred, test_ds, save_path, error_table, wandb)
        generate_confusion_matrix(y_true, y_pred, test_ds.class_names, save_path, wandb)
        save_results_local_and_wandb(args, test_ds, y_true, y_pred, save_path, accuracy, wandb)

        wandb.log({"misclassified_samples": error_table})

        # RETORNA OS DADOS BRUTOS PARA O MAIN
        return profiler.get_raw_data()

    except Exception as e:
        print("An error occurred during testing:", str(e))
        return None 
    finally:
        wandb.finish()
        del model 
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    models = ['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x']
    colors = ['LAB', 'HSV']
    datasets = {
        'test': 'Datasets/LIST/test',
        'cross_test': 'Datasets/CSIQ',
    }

    global_hardware_data = {
        m: {'ram': [], 'gpu': [], 'vram': [], 'power': []} for m in models
    }

    for folder_name, dataset_path in datasets.items():
        for color in colors:
            for model_name in models:
                family, version = extract_model_parts(model_name)

                hierarchical_path = f"runs/tested2/{family}/{version}/{folder_name}/{color}"
                experiment_id = f"{version}_{folder_name}_{color}"
                
                args = {
                    "base_path": hierarchical_path,
                    "experiment_name": experiment_id,
                    "dataset_path" : dataset_path,
                    "weights_path": f'runs/trained/{family}/{version}/{color}/best.pt', 
                    "model": model_name,
                    "image_mode": color,
                    "evaluation_folder": folder_name,
                    "hardware_evaluation": False
                }

                raw_data = test(args)
                
                if raw_data is not None:
                    global_hardware_data[model_name]['ram'].extend(raw_data['ram'])
                    global_hardware_data[model_name]['gpu'].extend(raw_data['gpu'])
                    global_hardware_data[model_name]['vram'].extend(raw_data['vram'])
                    global_hardware_data[model_name]['power'].extend(raw_data['power'])

    # --- PROCESSAMENTO FINAL GLOBAL POR MODELO ---
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

            family, version = extract_model_parts(model_name)
            model_dir = f"runs/tested2/{family}/{version}"
            os.makedirs(model_dir, exist_ok=True)
            
            summary_path = f"{model_dir}/global_hardware_metrics.yaml"
            
            with open(summary_path, 'w') as f:
                yaml.dump({model_name: summary}, f, default_flow_style=False, sort_keys=False)
                
            print(f"[{version}] Global hardware summary saved to: {summary_path}")