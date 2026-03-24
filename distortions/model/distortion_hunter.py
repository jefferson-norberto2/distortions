import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torchvision import models

class FFTLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 1. Aplica a FFT 2D nas duas últimas dimensões (H, W)
        # x shape: [Batch, 3, H, W] -> Saída Complexa
        fft = torch.fft.fft2(x, dim=(-2, -1))
        
        # 2. Shift: Move as baixas frequências (zeros) para o centro da imagem
        # Isso facilita para a CNN aprender padrões simétricos
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        
        # 3. Magnitude: Pega o valor absoluto (sqrt(real^2 + imag^2))
        # O resultado agora é float (Real)
        magnitude = fft_shifted.abs()
        
        # 4. Escala Logarítmica: O espectro tem valores muito altos no centro.
        # O log "achata" esses valores para um range que a rede consegue aprender.
        # Adicionamos 1e-8 para evitar log(0).
        magnitude = torch.log(magnitude + 1e-8)
        
        return magnitude

class SobelLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Kernels oficiais do Sobel (Horizontal e Vertical)
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]]).view(1, 1, 3, 3)

        # Repete para os 3 canais (R, G, B) para não perder cor
        self.register_buffer('Gx', sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer('Gy', sobel_y.repeat(3, 1, 1, 1))

    def forward(self, x):
        # Aplica a convolução separadamente em cada canal (groups=3)
        grad_x = F.conv2d(x, self.Gx, padding=1, groups=3)
        grad_y = F.conv2d(x, self.Gy, padding=1, groups=3)
        
        # Calcula a Magnitude do Gradiente (A "força" da borda)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        return magnitude
    
class DistortionHunter(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # 1. Instancia o filtro Sobel
        self.sobel = SobelLayer()
        
        self.fft = FFTLayer()
        
        # 2. Carrega uma rede base (ex: ResNet18 ou MobileNet)
        # IMPORTANTE: Altere a primeira camada para aceitar 6 canais em vez de 3
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # O truque: Expandir os pesos da primeira camada para aceitar 6 canais
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            # Copia os pesos originais para os primeiros 3 canais (RGB)
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Copia os pesos originais também para os canais de Sobel (inicialização esperta)
            new_conv.weight[:, 3:, :, :] = old_conv.weight

        self.backbone.conv1 = new_conv
        
        # Ajusta a saída final
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        # 1. Extrai as bordas
        edges = self.sobel(x)
                
        # 2. Concatena: Agora a rede vê a cor E a textura explicitamente
        # Dimensão final: [Batch, 6, H, W]
        combined = torch.cat([x, edges], dim=1)
        
        # 3. Passa pela rede
        return self.backbone(combined)