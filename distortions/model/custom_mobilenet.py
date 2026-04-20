import torch
from torchvision import models
from torch.nn import Module, Linear
import timm

class CustomMobileNet(Module):
    def __init__(self, 
                 num_classes: int, 
                 pre_trained: bool,
                 backbone: str = 'mobilenet_v3_large'
                 ):
        
        super(CustomMobileNet, self).__init__()

        self.backbone = self.get_backbone_and_weights(name_model=backbone, pre_trained=pre_trained)
        
        # Adjusting the classifier index based on the chosen architecture
        if 'v3' in backbone:
            num_ftrs = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = Linear(num_ftrs, num_classes)
        elif backbone == 'mobilenet_v2':
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = Linear(num_ftrs, num_classes)
        elif backbone == 'mobilenet_v1':
            # timm models expose the final linear layer directly as 'classifier'
            num_ftrs = self.backbone.classifier.in_features
            self.backbone.classifier = Linear(num_ftrs, num_classes)
    
    def get_backbone_and_weights(self, name_model: str, pre_trained: bool) -> Module:
        if name_model == 'mobilenet_v3_large':
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pre_trained else None
            backbone_model = models.mobilenet_v3_large(weights=weights)
        elif name_model == 'mobilenet_v3_small':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pre_trained else None
            backbone_model = models.mobilenet_v3_small(weights=weights)
        elif name_model == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pre_trained else None
            backbone_model = models.mobilenet_v2(weights=weights)
        elif name_model == 'mobilenet_v1':
            # Load MobileNetV1 from Hugging Face timm
            backbone_model = timm.create_model('mobilenetv1_100', pretrained=pre_trained)
        else:
            raise ValueError(f"Model error: {name_model}, choose: 'mobilenet_v3_large', 'mobilenet_v3_small', 'mobilenet_v2', 'mobilenet_v1'.")
        
        return backbone_model

    def forward(self, x):
        return self.backbone(x)

# --- Test Example ---
if __name__ == "__main__":
    # Instantiating the model with MobileNetV1 backbone
    model = CustomMobileNet(num_classes=7, pre_trained=True, backbone='mobilenet_v1')
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Output shape for MobileNetV1: {output.shape}") 
    # Expected: torch.Size([1, 5])