from torch import nn, randn
from torchvision import models

def get_resnet_model(name):
    # Dicionário para facilitar a seleção
    resnet_models = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
        'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
        'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
        'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT),
        'resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT),
    }
    
    if name not in resnet_models:
        raise ValueError(f"Modelo {name} não suportado.")
    
    model_func, weights = resnet_models[name]
    return model_func(weights=weights)

def get_mobilenet_model(name):
    mobilenet_models = {
        'mobilenet_v3_small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
        'mobilenet_v3_large': (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
    }
    
    if name not in mobilenet_models:
        raise ValueError(f"Modelo {name} não suportado.")
    
    model_func, weights = mobilenet_models[name]
    return model_func(weights=weights)

class Extractor(nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        # 1. Carrega o modelo completo
        if name.startswith('resnet'):
            self.backbone = get_resnet_model(name)
            if name in ['resnet18', 'resnet34']:
                self.feature_dim = 512
            else:
                self.feature_dim = 2048
        elif name.startswith('mobilenet'):
            self.backbone = get_mobilenet_model(name)
            self.feature_dim = 1000
        elif name.startswith('inception'):
            self.backbone = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Modelo {name} não suportado.")
            
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        # Agora retorna (Batch, 512) em vez de (Batch, 512, 1, 1)
        return self.backbone(x)

if __name__ == "__main__":
    # Teste rápido
    extractor = Extractor('mobilenet_v3_large')
    dummy_input = randn(1, 3, 512, 512)
    output = extractor(dummy_input)
    print(f"Shape da saída: {output.shape}") # Esperado: torch.Size([1, 512])