import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstrainedConv2d(nn.Conv2d):
    """
    Camada de Convolução Restrita (Bayar & Stamm).
    Força o kernel a agir como um filtro passa-alta adaptativo.
    Restrições:
    1. O peso central é fixado em -1.
    2. A soma de todos os pesos no kernel deve ser 0.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False):
        super(ConstrainedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.center = (kernel_size - 1) // 2

        # Cria uma máscara binária onde apenas o pixel central é 0
        self.register_buffer('mask', torch.ones(self.weight.size()))
        self.mask[:, :, self.center, self.center] = 0

    def forward(self, x):
        # 1. Zera o pixel central dos pesos atuais para cálculo da soma
        masked_weights = self.weight * self.mask
        
        # 2. Calcula a soma dos pesos restantes
        weight_sum = torch.sum(masked_weights, dim=(2, 3), keepdim=True)
        
        # 3. Normaliza para que a soma dos vizinhos seja 1 (evita divisão por zero com eps)
        normalized_weights = masked_weights / (weight_sum + 1e-6)
        
        # 4. Substitui o centro por -1. 
        # Agora a soma total do filtro é: 1 + (-1) = 0
        final_weights = normalized_weights - (1 - self.mask) # Subtrai 1 onde a máscara é 0 (no centro)

        return F.conv2d(x, final_weights, self.bias, self.stride, self.padding)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block para recalibrar a importância dos canais de ruído.
    Útil para diferenciar ruído de cor vs ruído de luminância.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TextureResBlock(nn.Module):
    """
    Bloco Residual modificado com SE-Block integrado.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(TextureResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) # Aplica atenção
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class NoiseClassificationNet(nn.Module):
    def __init__(self, num_classes=7):
        super(NoiseClassificationNet, self).__init__()
        
        # 1. Camada de Pré-processamento Restrito
        self.preprocessing = ConstrainedConv2d(3, 64, kernel_size=5, padding=2)
        
        # 2. Backbone (Feature Extractor)
        self.layer1 = TextureResBlock(64, 64, stride=1)
        self.layer2 = TextureResBlock(64, 128, stride=2)
        self.layer3 = TextureResBlock(128, 256, stride=2)
        self.layer4 = TextureResBlock(256, 512, stride=2)

        # 3. Classificador (Head) - AGORA DENTRO DA CLASSE
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.preprocessing(x) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pooling e Flatten
        x = self.avg_pool(x)

        # Flatten: [Batch, 512]
        x = x.view(x.size(0), -1) 
        
        # Classificação
        x = self.fc(x) 
        return x

# Exemplo de instancialização e teste
if __name__ == "__main__":
    # Configuração
    num_classes = 7 # Ex: Blur, Salt&Pepper, Poisson, Clean
    model = NoiseClassificationNet(num_classes=num_classes)
    
    # Simulação de entrada (Batch de 8 imagens, 299x299 RGB)
    dummy_input = torch.randn(8, 3, 299, 299)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") 
    print("Arquitetura carregada com sucesso.")