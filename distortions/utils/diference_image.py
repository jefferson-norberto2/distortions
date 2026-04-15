import cv2
import matplotlib.pyplot as plt
import numpy as np

def comparar_imagens_numeros(caminho_img1, caminho_img2):
    # 1. Carregar as imagens
    img1 = cv2.imread(caminho_img1)
    img2 = cv2.imread(caminho_img2)

    if img1 is None or img2 is None:
        print("Erro: Não foi possível carregar as imagens. Verifique os caminhos.")
        return

    # Garantir mesmo tamanho
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Converter de BGR (padrão do OpenCV) para RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 2. Converter os números para um formato que suporte valores negativos e maiores
    # (int16) para podermos fazer a conta matemática de subtração tranquilamente
    img1_calc = img1_rgb.astype(np.int16)
    img2_calc = img2_rgb.astype(np.int16)

    # 3. Calcular a diferença absoluta (transforma números negativos em positivos)
    diff = np.abs(img1_calc - img2_calc)

    # Separar os canais R (0), G (1) e B (2)
    diff_r = diff[:, :, 0]
    diff_g = diff[:, :, 1]
    diff_b = diff[:, :, 2]

    # 4. Calcular as médias de diferença (o quanto as cores variaram de 0 a 255)
    media_r = np.mean(diff_r)
    media_g = np.mean(diff_g)
    media_b = np.mean(diff_b)
    media_total = np.mean(diff)

    # 5. Calcular a porcentagem de diferença (baseado no máximo possível, que é 255)
    perc_r = (media_r / 255) * 100
    perc_g = (media_g / 255) * 100
    perc_b = (media_b / 255) * 100
    perc_total = (media_total / 255) * 100

    # 6. Calcular a quantidade exata de pixels que mudaram (pelo menos um pouco)
    # Como a imagem tem 3 canais, se somarmos onde a diferença > 0 e dividirmos por 3,
    # temos a noção de pixels afetados.
    mascara_diferentes = np.any(diff > 0, axis=-1)
    qtd_pixels_diff = np.sum(mascara_diferentes)
    total_pixels = img1.shape[0] * img1.shape[1]
    perc_pixels = (qtd_pixels_diff / total_pixels) * 100

    print("="*50)
    print("      RELATÓRIO NUMÉRICO DE DIFERENÇA RGB")
    print("="*50)
    print(f"Total de Pixels da Imagem: {total_pixels}")
    print(f"Pixels que sofreram alteração: {qtd_pixels_diff} ({perc_pixels:.2f}% da imagem)")
    print("-" * 50)
    print(f"Diferença Média de Cor (Geral): {media_total:.2f} níveis ({perc_total:.2f}%)")
    print(f"-> Canal Vermelho (R): Variou em média {media_r:.2f} níveis ({perc_r:.2f}%)")
    print(f"-> Canal Verde (G)   : Variou em média {media_g:.2f} níveis ({perc_g:.2f}%)")
    print(f"-> Canal Azul (B)    : Variou em média {media_b:.2f} níveis ({perc_b:.2f}%)")
    print("="*50)


if __name__ == '__main__':
    comparar_imagens_numeros('Datasets/Subset/train/4.png', 'Datasets/Originals/train/4.jpg')