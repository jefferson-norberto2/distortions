from torchvision import models
from torch.nn import Module, Linear

class CustomResNet(Module):
    def __init__(self, 
                 num_classes: int, 
                 backbone, 
                 weights
                 ):
        
        super(CustomResNet, self).__init__()
    
        self.backbone = backbone(weights=weights)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)

class CustomInception(Module):
    def __init__(self, num_classes, backbone, weights, training):
        super().__init__()
        self.backbone = backbone(weights=weights)
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