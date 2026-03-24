from torch import nn
import timm

# ==========================================
# 2. O Modelo (Swin Transformer)
# ==========================================
class SwinDistortionModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_name='swin_tiny_patch4_window7_224'):
        super().__init__()
        # Cria o modelo via timm
        # pretrained=True baixa os pesos do ImageNet (transfer learning)
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.model(x)