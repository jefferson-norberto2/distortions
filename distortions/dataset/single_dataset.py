from PIL import Image
from distortions.dataset.custom_dataset import CustomDataset

class SingleDataset(CustomDataset):
    def __init__(self, root, image_mode='RGB'):
        super().__init__(root)
        self.image_mode = image_mode

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img_rgb = Image.open(path).convert(self.image_mode)
        
        if self.transform_rgb:
            img_rgb = self.transform_rgb(img_rgb)
        
        return img_rgb, label