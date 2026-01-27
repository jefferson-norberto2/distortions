
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import Generator
import os

def is_train_val_dataset(datadir: str) -> bool:
	'''
	Checks if the dataset directory contains separate 'train' and 'val' subdirectories.

	Args:
		datadir (str): Path to the dataset directory.

	Returns:
		bool: True if both 'train' and 'val' subdirectories exist, False otherwise.
	'''
	train_dir = os.path.join(datadir, 'train')
	val_dir = os.path.join(datadir, 'val')
	return os.path.isdir(train_dir) and os.path.isdir(val_dir)

def _get_transform(im_size: int) -> transforms.Compose:
	'''
	Creates a transformation pipeline for image preprocessing.

	Args:
		im_size (int): Size to which images will be resized.
	Returns:
		transforms.Compose: Composed transformations.
	'''
	return transforms.Compose([
		transforms.CenterCrop(im_size),
		transforms.ToTensor(),
	])

def get_val_dataloader(data_dir: str, batch_size: int, im_size: int) -> tuple[DataLoader, datasets.ImageFolder]:
	'''
	Creates a validation dataloader from the dataset located at data_dir.
	
	Args:
		data_dir (str): Path to the dataset directory.
		batch_size (int): Number of samples per batch.
		im_size (int): Size to which images will be resized.

	Returns:
		DataLoader: Validation dataloader.
	'''
	transform = _get_transform(im_size)
	
	dataset = datasets.ImageFolder(root=data_dir, transform=transform)
	val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	return val_loader, dataset

def _train_val_loaders_from_dirs(data_dir: str, transform: transforms.Compose, batch_size: int) -> tuple[DataLoader, DataLoader]:
	'''
	Creates training and validation dataloaders from 'train' and 'val' subdirectories in data_dir.
	
	Args:
		data_dir (str): Path to the dataset directory.
		transform (transforms.Compose): Transformations to apply to the images.
		batch_size (int): Number of samples per batch.
	Returns:
		tuple[DataLoader, DataLoader]: Training and validation dataloaders.
	'''
	train_dir = os.path.join(data_dir, 'train')
	val_dir = os.path.join(data_dir, 'val')

	train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
	val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

	# Show dataset sizes
	print(f"📂 Train dataset size: {len(train_dataset)} images")
	print(f"📂 Validation dataset size: {len(val_dataset)} images")

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
 
	return train_loader, val_loader

def _train_val_loaders_from_split(data_dir: str, train_split: float, transform: transforms.Compose, batch_size: int) -> tuple[DataLoader, DataLoader]:
	'''
	Creates training and validation dataloaders by splitting the dataset located at data_dir.
	
	Args:
		data_dir (str): Path to the dataset directory.
		train_split (float): Proportion of the dataset to use for training
		transform (transforms.Compose): Transformations to apply to the images.
		batch_size (int): Number of samples per batch.
	Returns:
		tuple[DataLoader, DataLoader]: Training and validation dataloaders.
	'''	
	dataset = datasets.ImageFolder(root=data_dir, transform=transform)

	# Show dataset size
	print(f"📂 Dataset size: {len(dataset)} images")

	train_size = int(train_split * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=Generator().manual_seed(42))

	# Show split sizes
	print(f"🚂 Train set size: {len(train_dataset)} images")
	print(f"🧪 Validation set size: {len(val_dataset)} images")

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
 
	return train_loader, val_loader

def get_train_dataloaders(data_dir: str, batch_size: int, im_size: int) -> tuple[DataLoader, DataLoader]:
	'''
	Creates training and validation dataloaders from the dataset located at data_dir.
	
	Args:
		data_dir (str): Path to the dataset directory.
		train_split (float): Proportion of the dataset to use for training.
		batch_size (int): Number of samples per batch.
		im_size (int): Size to which images will be resized.
		is_training (bool): Flag indicating if the dataloaders are for training.

	Returns:
		tuple[DataLoader, DataLoader]: Training and validation dataloaders.
	'''
	transform = _get_transform(im_size)
	
	train_loader, val_loader = None, None
 
	if is_train_val_dataset(data_dir):
		train_loader, val_loader = _train_val_loaders_from_dirs(data_dir, transform, batch_size)
	else:
		train_loader, val_loader = _train_val_loaders_from_split(data_dir, 0.7, transform, batch_size)

	return train_loader, val_loader