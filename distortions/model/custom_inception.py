from torchvision import models
from torch.nn import Module, Linear

class CustomInception(Module):
    def __init__(self, num_classes, pre_treined, training):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pre_treined else None
        
        self.backbone = models.inception_v3(weights=weights, aux_logits=True)
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