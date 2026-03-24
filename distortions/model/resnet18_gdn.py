import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet

class GDN(nn.Module):
    def __init__(self, in_channels: int, inverse: bool = False, beta_min: float = 1e-6, gamma_init: float = 0.1):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.in_channels = in_channels

        self.beta_reparam = nn.Parameter(torch.ones(in_channels))
        self.gamma_reparam = nn.Parameter(
            torch.eye(in_channels) * gamma_init +
            torch.rand(in_channels, in_channels) * 1e-4
        )

    def forward(self, x):
        beta = F.softplus(self.beta_reparam) + self.beta_min
        gamma = F.softplus(self.gamma_reparam)
        gamma = gamma.view(self.in_channels, self.in_channels, 1, 1)

        norm_pool = F.conv2d(x ** 2, gamma, bias=beta)
        norm_pool = torch.sqrt(norm_pool)

        if self.inverse:
            return x * norm_pool
        else:
            return x / norm_pool

class GDNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(GDNBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.gdn1 = GDN(in_channels=planes) # GDN no meio do bloco
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.gdn2 = GDN(in_channels=planes) # GDN no final do bloco
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gdn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gdn2(out)

        return out

def resnet18_gdn(num_classes: int):
    # Instancia a ResNet usando nosso bloco com GDN. 
    # [2, 2, 2, 2] é a configuração padrão de blocos da ResNet-18.
    model = ResNet(GDNBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    
    # Substitui a ReLU inicial da rede por uma GDN (a conv1 de entrada gera 64 canais)
    model.relu = GDN(in_channels=64)
    
    return model