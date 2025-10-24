import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# importar biblioteca para listar imagens na pasta
import os

def gerar_ruido_rosa_2d(height, width, beta=2):
    """
    Gera ruído rosa 2D (para imagens) usando FFT.

    Args:
        height (int): Altura da imagem.
        width (int): Largura da imagem.
        beta (int): O expoente da lei de potência (Potência ~ 1/f^beta).
                    beta = 0 é ruído branco.
                    beta = 1 é "pink noise" 1D (em 2D fica "fraco").
                    beta = 2 é "pink noise" 2D (também chamado de "brown noise" 1D).
                    Este é o valor mais comum para texturas naturais.

    Returns:
        numpy.ndarray: Uma matriz 2D com o ruído, normalizada entre 0 e 1.
    """
    
    # 1. Criar um grid de frequências
    # Calcula as frequências para cada eixo
    freq_x = np.fft.fftfreq(width)
    freq_y = np.fft.fftfreq(height)
    
    # Cria um grid 2D com essas frequências
    fxx, fyy = np.meshgrid(freq_x, freq_y)
    
    # 2. Calcular a magnitude da frequência (distância do centro)
    # f = sqrt(fx^2 + fy^2)
    f_squared = fxx**2 + fyy**2
    
    # 3. Criar o filtro de amplitude (Amplitude ~ 1 / f^(beta/2))
    # Lidando com a divisão por zero no componente DC (f=0)
    # Definimos f[0,0] = 1.0 para evitar o erro, sua amplitude será 1.
    f_squared[0, 0] = 1.0
    
    # A Potência(f) ~ 1/f^beta
    # A Amplitude(f) ~ sqrt(Potência) ~ 1/f^(beta/2)
    amplitude_filtro = 1.0 / (f_squared ** (beta / 4.0)) # (f^2)^(beta/4) = f^(beta/2)
    
    # 4. Gerar ruído branco no domínio da frequência
    # Fases aleatórias (números complexos aleatórios)
    fase_real = np.random.normal(0, 1, (height, width))
    fase_imaginaria = np.random.normal(0, 1, (height, width))
    ruido_fft = (fase_real + 1j * fase_imaginaria)
    
    # 5. Aplicar o filtro de amplitude ao ruído
    fft_filtrado = ruido_fft * amplitude_filtro
    
    # 6. Aplicar a Transformada Inversa (iFFT)
    # Isso nos leva de volta ao domínio espacial (a imagem)
    ruido_espacial = np.fft.ifft2(fft_filtrado)
    
    # 7. Pegar a parte real e normalizar
    ruido_final = np.real(ruido_espacial)
    
    # Normaliza o ruído para o intervalo [0, 1]
    ruido_normalizado = (ruido_final - np.min(ruido_final)) / \
                        (np.max(ruido_final) - np.min(ruido_final))
    
    return ruido_normalizado

# --- Exemplo de Uso ---

IMG_SIZE = 512

noise_r = gerar_ruido_rosa_2d(IMG_SIZE, IMG_SIZE, beta=2)
noise_g = gerar_ruido_rosa_2d(IMG_SIZE, IMG_SIZE, beta=2)
noise_b = gerar_ruido_rosa_2d(IMG_SIZE, IMG_SIZE, beta=2)

# Empilha os canais de ruído
pink_noise_rgb = np.stack([noise_r, noise_g, noise_b], axis=-1)

root_path = '/home/jmn/dev/Datasets/IQA/LIVE/fnoise'

list_files = os.listdir(root_path)

for file_name in list_files:
    if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.bmp'):
        image_path = os.path.join(root_path, file_name)
        print(f"Processando imagem: {image_path}")

        # 1. Carregar a imagem RGB
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img, dtype=np.float32)

        # resize se necessário
        img_array = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)

        # 2. Normalizar a imagem para [0, 1]
        img_normalized = img_array / 255.0

        # 3. Adicionar o ruído rosa (ajustar a intensidade conforme necessário)
        intensity = 0.3  # Ajuste a intensidade do ruído
        noisy_img = img_normalized + intensity * pink_noise_rgb
        noisy_img = np.clip(noisy_img, 0, 1)  # Garantir que os valores estejam em [0, 1]

        # 4. Converter de volta para imagem e salvar/mostrar
        noisy_img_uint8 = (noisy_img * 255).astype(np.uint8)
        noisy_image = Image.fromarray(noisy_img_uint8)
        noisy_image.save(image_path)

