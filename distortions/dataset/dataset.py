
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import Generator

def get_dataloaders(data_dir="/home/jmn/host/dev/Datasets/IQA/ECSIQ/", train_split=0.8, image_shape=(256,256), batch_size=16):
    transform = transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader