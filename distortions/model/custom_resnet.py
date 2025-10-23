from torchvision import models
from torch.nn import Module, Linear

class CustomResNet(Module):
    def __init__(self, 
                 num_classes=7, 
                 backbone=models.resnet50, 
                 weights=models.ResNet50_Weights.IMAGENET1K_V2
                 ):
        
        super(CustomResNet, self).__init__()
    
        self.backbone = backbone(weights=weights)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)
