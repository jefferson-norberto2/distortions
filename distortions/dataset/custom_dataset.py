from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from abc import ABC, abstractmethod

class CustomDataset(Dataset, ABC):
    def __init__(self, root: str, image_mode='RGB', image_size=(512, 512)):
        self.samples = []
        self.class_names = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.image_mode = image_mode
        self.image_size = image_size

        for cls_name in self.class_names:
            cls_path = Path(root) / cls_name
            for img_path in cls_path.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))
        
        self.transform_rgb = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 

        self.transform_changed = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self): 
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx):
        pass