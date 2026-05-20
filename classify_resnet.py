from distortions.model.custom_resnet import CustomResNet
import torch
import yaml
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

model = CustomResNet(num_classes=7, pre_trained=False, backbone='resnet50')
model.load_state_dict(torch.load('/media/jmn/Removable Disk/runs/trained/resnet/V1/HSV/best.pt' , map_location='cpu', weights_only=True))
model.eval()

root_dir = '/media/jmn/Removable Disk/database/Macro'
args_path = '/media/jmn/Removable Disk/runs/trained/resnet/V1/HSV/args.yaml'
dataset_save_path = 'Datasets/Wild_v2'

os.makedirs(dataset_save_path, exist_ok=True)

with open(args_path, 'r') as file:
    data = yaml.safe_load(file)

class_names = data['class_names']
ids = {i: name for i, name in enumerate(class_names)}

for class_name in class_names:
    os.makedirs(os.path.join(dataset_save_path, class_name), exist_ok=True)

for img_name in tqdm(os.listdir(root_dir)):
    img_path = os.path.join(root_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    im = img.resize((512, 512))
    img.thumbnail((1000, 1000))
    img_tensor = transforms.ToTensor()(im).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = ids[predicted.item()]
        
    save_path = os.path.join(dataset_save_path, predicted_class, img_name)
    img.save(save_path)
