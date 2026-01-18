import os
import shutil
from torch.utils.data import Subset

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import Generator, tensor, stack
from torch.nn.functional import pad

def pad_collate(batch):
    images, labels = zip(*batch)

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    for img in images:
        h, w = img.shape[1], img.shape[2]

        pad_w = max_w - w
        pad_h = max_h - h
        img = pad(img, (0, pad_w, 0, pad_h))  
        padded_images.append(img)

    return stack(padded_images), tensor(labels)


def get_dataloaders(data_dir: str, train_split: float, batch_size: int) -> tuple[DataLoader, DataLoader]:
	'''
	Creates training and validation dataloaders from the dataset located at data_dir.
	
	Args:
		data_dir (str): Path to the dataset directory.
		train_split (float): Proportion of the dataset to use for training.
		batch_size (int): Number of samples per batch.

	Returns:
		tuple[DataLoader, DataLoader]: Training and validation dataloaders.
	'''
	transform = transforms.Compose([
        #transforms.Resize(299),
		transforms.ToTensor(),
	])

	dataset = datasets.ImageFolder(root=data_dir, transform=transform)

	# Show dataset size
	print(f"📂 Dataset size: {len(dataset)} images")

	train_size = int(train_split * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=Generator().manual_seed(42))

	# Show split sizes
	print(f"🚂 Train set size: {len(train_dataset)} images")
	print(f"🧪 Validation set size: {len(val_dataset)} images")

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
	val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

	return train_loader, val_loader, train_dataset, val_dataset


def save_dataset_to_disk(subset: Subset, output_dir: str, split_name: str):
    """
    Copia as imagens de um Subset para uma pasta estruturada por classes.
    """
    # Recupera o dataset original (ImageFolder)
    original_dataset = subset.dataset
    classes = original_dataset.classes
    
    print(f"📦 Salvando imagens de {split_name} em: {output_dir}")

    for idx in subset.indices:
        # Pega o caminho original do arquivo e o índice da classe
        img_path, class_idx = original_dataset.samples[idx]
        class_name = classes[class_idx]
        
        # Cria a estrutura de pastas: output/split/classe/
        target_folder = os.path.join(output_dir, split_name, class_name)
        os.makedirs(target_folder, exist_ok=True)
        
        # Copia o arquivo original (mais rápido e mantém a qualidade original)
        shutil.copy(img_path, os.path.join(target_folder, os.path.basename(img_path)))

# --- Exemplo de uso integrado à sua função ---

if __name__ == "__main__":

	# 1. Gera os subsets usando sua função modificada para retornar os datasets
	# (Dica: é melhor retornar o train_dataset e val_dataset antes de virarem Loaders)
	train_loader, val_loader, train_data, val_data = get_dataloaders("./data/ECSIQ", train_split=0.8, batch_size=16)

	# 2. Salva fisicamente
	output_path = "./data/ECSIQ_YOLO"
	save_dataset_to_disk(train_data, output_path, "train")
	save_dataset_to_disk(val_data, output_path, "val")