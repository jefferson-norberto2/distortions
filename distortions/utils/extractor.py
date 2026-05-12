from torch import nn, randn
from torchvision import models
import timm

def get_resnet_model(name):
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

def get_mobilenet_model(name: str):
    mobilenet_models = {
        'mobilenet_v2': (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
        'mobilenet_v3_small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
        'mobilenet_v3_large': (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
    }
    
    if name.lower() == 'mobilenet_v1':
        backbone_model = timm.create_model('mobilenetv1_100', pretrained=True)
        return backbone_model
    
    if name.lower() not in mobilenet_models:
        raise ValueError(f"Modelo {name} não suportado.")
    
    model_func, weights = mobilenet_models[name]
    return model_func(weights=weights)

class Extractor(nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        
        # 1. Carrega o modelo de forma normal
        if name.startswith('resnet'):
            self.backbone = get_resnet_model(name)
        elif name.startswith('mobilenet'):
            self.backbone = get_mobilenet_model(name)
        elif name.startswith('inception'):
            self.backbone = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        else:
            raise ValueError(f"Modelo {name} não suportado.")

        # 2. Descobre a dimensão automaticamente e remove a camada final
        self.feature_dim = self._extract_features_and_remove_head()

    def _extract_features_and_remove_head(self):
        """
        Inspeciona o modelo para descobrir o tamanho do vetor de características.
        """
        feature_dim = None

        # Para modelos da família ResNet e Inception
        if hasattr(self.backbone, 'fc'):
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        # Para modelos da família MobileNet (e VGG, DenseNet, etc)
        elif hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Sequential):
                # Como vamos remover o bloco todo, a dimensão que vai sobrar 
                # é a que ENTRARIA na primeira camada Linear desse bloco.
                for layer in self.backbone.classifier:
                    if isinstance(layer, nn.Linear):
                        feature_dim = layer.in_features
                        break # Achou a primeira, pode parar
            else:
                feature_dim = self.backbone.classifier.in_features
                
            self.backbone.classifier = nn.Identity()
            
        else:
            raise NotImplementedError("Estrutura da camada final desconhecida para este modelo.")
            
        return feature_dim

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # Teste rápido
    extractor = Extractor('resnet50')
    dummy_input = randn(1, 3, 512, 512)
    output = extractor(dummy_input)
    print(f"Shape da saída: {output.shape}") # Esperado: torch.Size([1, 512])