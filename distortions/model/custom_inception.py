from torchvision import models
from torch.nn import Module, Linear

class CustomInception(Module):
    def __init__(self, num_classes, pre_treined, training=True):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pre_treined else None
        
        # AJUSTE 1: Desativamos aux_logits aqui. 
        # Isso evita o erro de dimensão com inputs de 128x128.
        self.backbone = models.inception_v3(weights=weights, aux_logits=False)
        self.training_mode = training # Renomeei para evitar conflito com self.training do nn.Module

        # Troca FC principal
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = Linear(num_ftrs, num_classes)

        # Não precisamos mais mexer no AuxLogits.fc porque ele não existe mais.

    def forward(self, x):
        # O retorno do inception muda quando aux_logits=False.
        # Ele retorna direto o Tensor, e não mais o objeto InceptionOutputs.
        out = self.backbone(x)
        
        return out