from PIL import Image
from distortions.dataset.custom_dataset import CustomDataset

class SingleDataset(CustomDataset):
    def __init__(self, root, image_mode='RGB', image_size=(512, 512)):
        super().__init__(root, image_mode=image_mode, image_size=image_size)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert(self.image_mode)
        
        if self.image_mode == 'RGB':
            img = self.transform_rgb(img)
        else:
            img = self.transform_changed(img)
        
        return img, label        