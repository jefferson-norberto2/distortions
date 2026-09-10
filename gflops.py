import torch
import torchvision.models as models
from fvcore.nn import FlopCountAnalysis, flop_count_table 

# 1. Configurar o dispositivo para a sua RTX 4050
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rodando em: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

# 2. Carregar o seu modelo de classificação de imagem e mover para a GPU
modelo = models.resnet50(weights=None).to(device)
modelo.eval()  # Modo de avaliação desativa Dropout e BatchNorm

# 3. Criar uma imagem fictícia (Tensor) condizente com a entrada do seu modelo
# Padrão para classificação: [Batch_Size, Canais_RGB, Altura, Largura]
tamanho_entrada = (1, 3, 224, 224)
imagem_teste = torch.randn(tamanho_entrada).to(device)

# 4. Analisar os FLOPs usando o fvcore com gradientes desativados
with torch.no_grad():
    analisador = FlopCountAnalysis(modelo, imagem_teste)
    
    # O fvcore calcula por padrão os MACs (Multiply-Accumulate) como 1 unidade
    total_macs = analisador.total()
    total_flops = total_macs * 2  # 1 MAC = 1 multiplicação + 1 soma (2 operações)

# 5. Converter os resultados para a escala Giga (dividir por 1e9)
gmacs = total_macs / 1e9
gflops = total_flops / 1e9

# Exibir os resultados estruturados
print("-" * 40)
print(f"Resolução da imagem de entrada: {tamanho_entrada[2]}x{tamanho_entrada[3]}")
print(f"Total de GMACs por imagem: {gmacs:.3f} GMACs")
print(f"Total de GFLOPs por imagem: {gflops:.3f} GFLOPs")
print("-" * 40)

# BÔNUS: Imprime quais camadas estão consumindo mais processamento
print("\nDetalhamento por Camada (Breakdown):")
print(flop_count_table(analisador, max_depth=2))