from enum import Enum
from torch.nn import Module, Linear
from torchvision import models

class CustomResNet(Module):
    def __init__(self, 
                 num_classes: int, 
                 pre_treined: bool,
                 backbone: str = 'resnet_50'
                 ):
        
        super(CustomResNet, self).__init__()
    
        self.backbone = self.get_backbone_and_weights(name_model=backbone, pre_trained=pre_treined)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = Linear(num_ftrs, num_classes)
    
    def get_backbone_and_weights(self, name_model: str, pre_trained: bool) -> Module:
        if name_model == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pre_trained else None
            back = models.resnet18(weights=weights)
        elif name_model == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pre_trained else None
            back = models.resnet34(weights=weights)
        elif name_model == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pre_trained else None
            back = models.resnet50(weights=weights)
        elif name_model == 'resnet101':
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pre_trained else None
            back = models.resnet101(weights=weights)
        elif name_model == 'resnet152':
            weights = models.ResNet152_Weights.IMAGENET1K_V2 if pre_trained else None
            back = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Model error: {name_model}, choose: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.")
        return back

    def forward(self, x):
        return self.backbone(x)

