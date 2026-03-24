from PIL import Image
from distortions.dataset.custom_dataset import CustomDataset

class DualDataset(CustomDataset):
    def __init__(self, root):
        super().__init__(root)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        img_rgb = Image.open(path).convert('RGB')
        img_hsv = img_rgb.convert('HSV')

        if self.transform_rgb and self.transform_hsv:
            img_rgb, img_hsv = self.transform_rgb(img_rgb), self.transform_hsv(img_hsv)
        
        return img_rgb, img_hsv, label