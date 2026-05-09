from PIL import Image
from distortions.dataset.custom_dataset import CustomDataset

class DualDataset(CustomDataset):
    def __init__(self, root='path', image_mode='HSV', image_size=(256, 256)):
        super().__init__(root, image_mode=image_mode, image_size=image_size)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        img_rgb = Image.open(path).convert('RGB')

        if self.image_mode == 'HSV':
            img_changed = img_rgb.convert('HSV')
        elif self.image_mode == 'LAB':
            img_changed = img_rgb.convert('LAB')

        if self.transform_rgb and self.transform_changed:
            img_rgb, img_changed = self.transform_rgb(img_rgb), self.transform_changed(img_changed)
        
        return img_rgb, img_changed, label