from enum import Enum
from torch.nn import Module, Linear
from torchvision import models

class ModelArchitecture(Enum):
    RESNET_18 = "resnet_18"
    RESNET_34 = "resnet_34"
    RESNET_50 = "resnet_50"
    RESNET_101 = "resnet_101"
    RESNET_152 = "resnet_152"
    INCEPTION_V3 = "inception_v3"
    NOISE_NET = "noise_net"  
    
class CustomResNet(Module):
    def __init__(self, 
                 num_classes: int, 
                 backbone: ModelArchitecture,
                 pretrained: bool
                 ):
        
        super(CustomResNet, self).__init__()
    
        self.backbone = self.__get_backbone_and_weights(model_type=backbone, pretrained=pretrained)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)
    
    def __get_backbone_and_weights(self, model_type=ModelArchitecture.RESNET_50, pretrained=True) -> models.ResNet:
        if model_type == ModelArchitecture.RESNET_18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            back = models.resnet18(weights=weights)
        if model_type == ModelArchitecture.RESNET_34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            back = models.resnet34(weights=weights)
        elif model_type == ModelArchitecture.RESNET_50:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            back = models.resnet50(weights=weights)
        elif model_type == ModelArchitecture.RESNET_101:
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            back = models.resnet101(weights=weights)
        elif model_type == ModelArchitecture.RESNET_152:
            weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            back = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Model error: {model_type}. Choose: {list(ModelArchitecture)}.")
        return back

class CustomInception(Module):
    def __init__(self, num_classes: int, pre_trained: bool, training: bool):
        super().__init__()
        self.backbone = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1 if pre_trained else None)
        self.training = training

        # troca FC principal
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = Linear(num_ftrs, num_classes)

        # troca FC da cabeça auxiliar
        if self.backbone.AuxLogits is not None:
            num_aux = self.backbone.AuxLogits.fc.in_features
            self.backbone.AuxLogits.fc = Linear(num_aux, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        if self.training:
            return out.logits, out.aux_logits
        return out