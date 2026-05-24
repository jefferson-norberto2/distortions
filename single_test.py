from distortions.model.custom_mobilenet import CustomMobileNet
import torch
import os
from PIL import Image
from torchvision import transforms

images = os.listdir("Datasets/HRIQ/blur")

print(f"Found {len(images)} images in the blur folder.")

# Carrega o modelo treinado
model_path = "/run/media/jmn/Removable Disk/runs/trained/mobilenet/V1/HSV/best.pt"
model = CustomMobileNet(num_classes=7, pre_trained=False, backbone="mobilenet_v1").to("cpu")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


for img_name in images:
    img_path = os.path.join("Datasets/HRIQ/blur", img_name)
    image = Image.open(img_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        # filtered output to only consider 0, 1, 2, 6
        output = output[:, [0, 1, 2, 6]]
        predicted_class = torch.argmax(output, dim=1).item()
    
    print(f"Image: {img_name}, Predicted Class: {predicted_class}")