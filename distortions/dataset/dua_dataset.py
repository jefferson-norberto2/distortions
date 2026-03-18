from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DualDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.class_names = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        for cls_name in self.class_names:
            cls_path = Path(root) / cls_name
            for img_path in cls_path.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))
        default_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform = transform or default_transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_rgb = Image.open(path).convert('RGB')
        img_hsv = img_rgb.convert('HSV')
        if self.transform:
            return self.transform(img_rgb), self.transform(img_hsv), label
        return img_rgb, img_hsv, label