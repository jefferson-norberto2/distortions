import torch
from torchvision import models
from torch.nn import Module, Linear

# 2. Sua classe MobileNet modificada
class CustomMobileNetV3(Module):
    def __init__(self, 
                 num_classes: int, 
                 pre_trained: bool,
                 backbone: str = 'mobilenet_v3_large'
                 ):
        
        super(CustomMobileNetV3, self).__init__()

        self.backbone = self.get_backbone_and_weights(name_model=backbone, pre_trained=pre_trained)
        
        num_ftrs = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = Linear(num_ftrs, num_classes)
    
    def get_backbone_and_weights(self, name_model: str, pre_trained: bool) -> Module:
        if name_model == 'mobilenet_v3_large':
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pre_trained else None
            back = models.mobilenet_v3_large(weights=weights)
        elif name_model == 'mobilenet_v3_small':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pre_trained else None
            back = models.mobilenet_v3_small(weights=weights)
        else:
            raise ValueError(f"Model error: {name_model}, choose: 'mobilenet_v3_large', 'mobilenet_v3_small'.")
        return back

    def forward(self, x):
        return self.backbone(x)