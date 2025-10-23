
from torchvision import transforms
import torch
from distortions.model.custom_resnet import CustomResNet

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomResNet(num_classes=7)

    model = model.to(device)

    model.load_state_dict(torch.load("best_distortions_9_renet50.pth", map_location=device))

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()




