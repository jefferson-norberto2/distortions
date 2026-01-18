from dataclasses import dataclass
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Data class for loaders and class weights
@dataclass
class TrainLoader:
    dataset_name: str
    dataloaders: dict
    dataset_sizes: dict
    class_names: list
    class_weights: torch.Tensor
    train_dataset: datasets.ImageFolder
    val_dataset: datasets.ImageFolder
    train_distribution: dict
    val_distribution: dict

@dataclass
class TestLoader:
    dataset_name: str
    dataloader: DataLoader
    dataset_size: int
    class_names: list

def get_train_data_loader(root_dir: str, batch_size=32, input_size=299) -> TrainLoader:
    """
    Create DataLoaders for training and validation datasets using ImageFolder.
    Also calculates class weights to handle class imbalance.

    Args:
        root_dir (str): Root directory containing 'train' and 'val' subdirectories.
        batch_size (int): Batch size for DataLoaders.
        input_size (int): Size to which images will be cropped.

    Returns:
        dict: A dictionary containing DataLoaders, dataset sizes, class names, and class weights
                {
                    "dataloaders": dataloaders,
                    "dataset_sizes": dataset_sizes,
                    "class_names": class_names,
                    "class_weights": class_weights,
                    "train_dataset": train_dataset,
                    "val_dataset": val_dataset,
                    "train_distribution": train_distribution,
                    "val_distribution": val_distribution
                }
    """
    dataset_name = root_dir.strip('/').split('/')[-1]
    # 1. Transforms definition using normalization like ImageNet
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Loading Datasets using ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(root_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # 3. Automatic Calculation of Class Weights for Imbalance Handling
    # Since you have fewer images in the 'src' (original) class, we need to give it more weight.
    
    train_dataset = image_datasets['train']
    val_dataset = image_datasets['val']
    
    # Counting samples per train class
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    train_distribution = dict(zip(train_dataset.classes, class_counts))
    print(f"Train distribution: {train_distribution}")
    
    # Counting samples per val class (for information)
    val_class_counts = [0] * len(val_dataset.classes)
    for _, label in val_dataset.samples:
        val_class_counts[label] += 1
    
    val_distribution = dict(zip(val_dataset.classes, val_class_counts))
    print(f"Validation distribution: {val_distribution}")
    
    # Weight calculation: Inverse of frequency
    # Weight = Total_samples / (Num_classes * Samples_per_class)
    count_array = np.array(class_counts)
    total_samples = count_array.sum()
    n_classes = len(class_counts)
    
    class_weights = total_samples / (n_classes * count_array)
    class_weights = torch.FloatTensor(class_weights)
    
    print(f"Calculated weights for Loss: {class_weights}")

    # 4. Creating DataLoaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    loader = {
        "dataset_name": dataset_name,
        "dataloaders": dataloaders,
        "dataset_sizes": dataset_sizes,
        "class_names": class_names,
        "class_weights": class_weights,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_distribution": train_distribution,
        "val_distribution": val_distribution
    }

    return TrainLoader(**loader)

def get_test_data_loader(root_dir: str, batch_size=32, input_size=299) -> TestLoader:
    """
    Create DataLoader for test dataset using ImageFolder.

    Args:
        root_dir (str): Root directory containing the test dataset.
        batch_size (int): Batch size for DataLoader.
        input_size (int): Size to which images will be cropped.

    Returns:
        DataLoader: DataLoader for the test dataset.
    """
    # Transforms definition using normalization like ImageNet
    data_transforms = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    # Loading Dataset using ImageFolder
    test_dataset = datasets.ImageFolder(root_dir, data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    
    loader_info = TestLoader(
        dataset_name=root_dir.strip('/').split('/')[-1],
        dataloader=test_loader,
        dataset_size=len(test_dataset),
        class_names=test_dataset.classes
    )

    return loader_info

if __name__ == "__main__":
    # Ajuste o caminho para a raiz do seu dataset
    ROOT_PATH = "/home/jmn/dev/python/distortions/data/CSIQ_Noise/" 
    
    loaders = get_train_data_loader(ROOT_PATH, batch_size=32, input_size=299)
    
    print(f"\nClasses detectadas: {loaders['class_names']}")
    print(f"Tamanho do treino: {loaders['dataset_sizes']['train']}")
    print(f"Tamanho da validação: {loaders['dataset_sizes']['val']}")
    
    # Exemplo de como usar os pesos na Loss Function:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = loaders['class_weights'].to(device)
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    print("\nCriterion configurado com pesos para balanceamento!")