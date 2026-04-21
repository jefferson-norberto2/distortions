import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class YOLODataset(Dataset):
    def __init__(self, root_dir: str, target_size: int = 512, image_mode: str = 'RGB'):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.image_mode = image_mode
        self.samples = []
        
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for class_name in self.class_names:
            class_path = self.root_dir / class_name
            for img_path in class_path.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        
        # Load image (OpenCV reads in BGR format by default)
        image_bgr = cv2.imread(image_path)
        
        # Resize image
        image_resized = cv2.resize(image_bgr, (self.target_size, self.target_size))

        if self.image_mode == 'LAB':
            # OpenCV maps 8-bit LAB channels to 0-255
            image_lab = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
            image_tensor = torch.from_numpy(image_lab).permute(2, 0, 1).float() / 255.0

        elif self.image_mode == 'HSV':
            image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
            
            # Convert to float32 to perform channel-wise division
            image_float = image_hsv.astype(np.float32)
            
            # Channel 0 (Hue) is 0-179 in OpenCV uint8. Channels 1 and 2 are 0-255.
            image_float[:, :, 0] /= 179.0
            image_float[:, :, 1] /= 255.0
            image_float[:, :, 2] /= 255.0
            
            image_tensor = torch.from_numpy(image_float).permute(2, 0, 1)

        else:
            # Default to RGB
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        
        return image_tensor, label