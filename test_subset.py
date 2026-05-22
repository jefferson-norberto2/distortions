import os
import torch
import wandb
import gc

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from distortions.dataset.single_dataset import SingleDataset
from distortions.model.custom_mobilenet import CustomMobileNet
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


def test_subset(args: dict):
    """
    Testa modelo treinado com 7 classes em dataset com subset de 4 classes.
    """
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
        
        # Carrega dataset com subset de classes
        test_ds = SingleDataset(args['dataset_path'], image_mode=args['image_mode'], image_size=(args['imgsz'], args['imgsz']))
        subset_classes = args.get('subset_classes', None)
        
        if subset_classes:
            print(f"Using subset classes: {subset_classes}")
            test_ds.update_classes(subset_classes)
        
        print(f"Dataset classes: {test_ds.class_names}")
        print(f"Total samples: {len(test_ds)}")
        
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
        
        # Carrega modelo treinado com 7 classes
        num_classes_trained = args.get('num_classes_trained', 7)
        print(f"Loading model trained with {num_classes_trained} classes...")
        
        model = CustomMobileNet(num_classes=num_classes_trained, pre_trained=False, backbone=args['model']).to(device)
        model.load_state_dict(torch.load(args['weights_path'], map_location=device, weights_only=True))
        model.eval()

        error_table = wandb.Table(columns=["Image", "True Class", "Predicted Class"])

        v_acc, v_total = 0, 0
        y_true, y_pred = [], []
        
        profiler = HardwareProfiler(device_index=0)

        print("Running GPU warm-up...")
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        for _ in range(10):
            _ = model(dummy_input)
        
        # Mapeamento de índices: classe do subset -> probabilidades do modelo treinado
        class_to_trained_idx = {}
        all_classes_trained = args.get('all_classes_trained', test_ds.class_names)
        
        for subset_class_name in test_ds.class_names:
            if subset_class_name in all_classes_trained:
                class_to_trained_idx[subset_class_name] = all_classes_trained.index(subset_class_name)
        
        print(f"Class mapping: {class_to_trained_idx}")
        
        with torch.no_grad():
            pbar_val = tqdm(test_loader, desc=f"Testing {args['experiment_name']}: ")
            for x_rgb, labels in pbar_val:
                x_rgb, labels = x_rgb.to(device), labels.to(device)
                outputs = model(x_rgb)
                
                # Filtra apenas as classes do subset
                subset_class_indices = [class_to_trained_idx[cn] for cn in test_ds.class_names]
                filtered_outputs = outputs[:, subset_class_indices]
                
                preds = filtered_outputs.argmax(1)
                
                profiler.sample()
                
                v_acc += (preds == labels).sum().item()
                v_total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        accuracy = v_acc / v_total * 100
        args['accuracy'] = accuracy
        print(f"Test Accuracy: {accuracy:.2f}%")

        # Salva métricas de hardware
        hw_metrics = profiler.save_to_yaml(save_path)
        wandb.log({"hardware_averages": hw_metrics})

        check_predictions(y_true, y_pred, test_ds, save_path, error_table, wandb)
        generate_confusion_matrix(y_true, y_pred, test_ds.class_names, save_path, wandb)
        save_results_local_and_wandb(args, test_ds, y_true, y_pred, save_path, accuracy, wandb)

        wandb.log({"misclassified_samples": error_table})

        return profiler.get_raw_data()

    except Exception as e:
        print("An error occurred during testing:", str(e))
        import traceback
        traceback.print_exc()
        return None 
    finally:
        wandb.finish()
        del model 
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Configure aqui os parâmetros do seu teste
    args = {
        "base_path": "runs/tested/mobilenet/V1/subset_test",
        "experiment_name": "mobilenet_v1_subset_4classes",
        "dataset_path": "Datasets/WILD",  # MUDE ISSO
        "imgsz": 900,
        "weights_path": '/run/media/jmn/Removable Disk/runs/trained/mobilenet/V1/HSV/best.pt',  # VERIFIQUE O CAMINHO
        "model": "mobilenet_v1",
        "image_mode": "HSV",
        "evaluation_folder": "subset_test",
        "num_classes_trained": 7,  # Modelo foi treinado com 7 classes
        "subset_classes": ["awgn", "blur", "contrast", "src"],  # MUDE ISSO - 4 classes do seu subset
        "all_classes_trained": ["awgn", "blur", "contrast", "fnoise", "jpeg", "jpeg2000", "src"]  # MUDE ISSO - todas as 7 classes
    }

    raw_data = test_subset(args)
    
    if raw_data:
        print("\n=== Test completed successfully ===")
