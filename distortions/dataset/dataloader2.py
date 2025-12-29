import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def get_data_loaders(root_dir, batch_size=32, input_size=299):
    """
    Cria DataLoaders para a estrutura padrão de pastas:
    root_dir/train/class_name
    root_dir/val/class_name
    """
    
    # 1. Definição das Transformações
    # ATENÇÃO: Para classificação de ruído, evitamos Resize com interpolação
    # se a imagem já estiver no tamanho ou maior. O ideal é Crop.
    # Mas como você pediu 299x299 explícito, usaremos Resize para garantir.
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)), # Garante 299x299
            transforms.ToTensor(),
            # Normalização (opcional, mas recomendada para convergência)
            # Médias e Desvios padrão do ImageNet são um bom ponto de partida
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Carregamento dos Datasets usando ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(root_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # 3. Cálculo Automático de Pesos para o Desbalanceamento (Class Weights)
    # Como você tem menos imagens na classe 'src' (originais), precisamos dar mais peso a ela.
    
    train_dataset = image_datasets['train']
    
    # Contagem de amostras por classe
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
        
    print(f"Distribuição das classes no Treino: {dict(zip(train_dataset.classes, class_counts))}")

    # Cálculo do peso: Inverso da frequência
    # Peso = N_total / (N_classes * N_amostras_da_classe)
    count_array = np.array(class_counts)
    total_samples = count_array.sum()
    n_classes = len(class_counts)
    
    class_weights = total_samples / (n_classes * count_array)
    class_weights = torch.FloatTensor(class_weights)
    
    print(f"Pesos calculados para Loss: {class_weights}")

    # 4. Criação dos DataLoaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names, class_weights

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Ajuste o caminho para a raiz do seu dataset
    ROOT_PATH = "/home/jmn/dev/python/distortions/data/CSIQ_Noise/" 
    
    loaders, sizes, classes, weights = get_data_loaders(ROOT_PATH, batch_size=32, input_size=299)
    
    print(f"\nClasses detectadas: {classes}")
    print(f"Tamanho do treino: {sizes['train']}")
    print(f"Tamanho da validação: {sizes['val']}")
    
    # Exemplo de como usar os pesos na Loss Function:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = weights.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    print("\nCriterion configurado com pesos para balanceamento!")