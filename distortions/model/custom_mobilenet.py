import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import Module, Linear

# 1. Definimos a camada Sobel (Fixa, não treinável)
class SobelLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Kernels padrão do filtro Sobel
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]]).view(1, 1, 3, 3)

        # Registra como buffer (salva no state_dict, mas não treina)
        # Repetimos para 3 canais para aplicar em R, G e B independentemente
        self.register_buffer('Gx', sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer('Gy', sobel_y.repeat(3, 1, 1, 1))

    def forward(self, x):
        # groups=3 garante que o filtro seja aplicado canal por canal separadamente
        grad_x = F.conv2d(x, self.Gx, padding=1, groups=3)
        grad_y = F.conv2d(x, self.Gy, padding=1, groups=3)
        # Calcula a magnitude do gradiente (a força da borda)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

# 2. Sua classe MobileNet modificada
class CustomMobileNetV3(Module):
    def __init__(self, 
                 num_classes: int, 
                 pre_trained: bool,
                 backbone: str = 'mobilenet_v3_large'
                 ):
        
        super(CustomMobileNetV3, self).__init__()
    
        # Instancia o filtro Sobel
        self.sobel = SobelLayer()

        self.backbone = self.get_backbone_and_weights(name_model=backbone, pre_trained=pre_trained)
        
        # --- AQUI ESTÁ A MÁGICA (Adaptação da primeira camada) ---
        # A primeira camada do MobileNetV3 está dentro de features[0][0]
        # Ela é originalmente Conv2d(3, 16, kernel=3, stride=2...)
        first_conv_layer = self.backbone.features[0][0]
        
        # Criamos uma nova conv que aceita 6 canais (3 RGB + 3 Sobel)
        new_conv = nn.Conv2d(
            in_channels=6, # Mudamos de 3 para 6
            out_channels=first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=False # MobileNet geralmente não usa bias antes do BatchNorm
        )

        # Se for pré-treinado, precisamos aproveitar os pesos!
        if pre_trained:
            with torch.no_grad():
                # Copia os pesos originais para os primeiros 3 canais (RGB)
                new_conv.weight[:, :3, :, :] = first_conv_layer.weight
                # Para os novos 3 canais (Sobel), copiamos os mesmos pesos como inicialização
                # Isso ajuda a rede a não começar "cega" nesses canais
                new_conv.weight[:, 3:, :, :] = first_conv_layer.weight
        
        # Substitui a camada antiga pela nova
        self.backbone.features[0][0] = new_conv
        # ---------------------------------------------------------

        # Troca o classificador final (como você já fazia)
        num_ftrs = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = Linear(num_ftrs, num_classes)
    
    def get_backbone_and_weights(self, name_model: str, pre_trained: bool) -> Module:
        if name_model == 'mobilenet_v3_large':
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pre_trained else None
            back = models.mobilenet_v3_large(weights=weights)
        elif name_model == 'mobilenet_v3_small':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pre_trained else None
            back = models.mobilenet_v3_small(weights=weights)
        else:
            raise ValueError(f"Model error: {name_model}, choose: 'mobilenet_v3_large', 'mobilenet_v3_small'.")
        return back

    def forward(self, x):
        # 1. Extrai as bordas/textura
        edges = self.sobel(x)
        
        # 2. Concatena (Junta) as imagens: RGB + Bordas
        # Dimensão de entrada: [Batch, 3, H, W]
        # Dimensão combinada: [Batch, 6, H, W]
        combined = torch.cat([x, edges], dim=1)
        
        # 3. Passa para o backbone (que agora aceita 6 canais)
        return self.backbone(combined)