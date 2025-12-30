import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import  cv2

from distortions.utils.pink_noise import gerar_ruido_rosa_2d

def mapa_intensidade_blur(height, width, min_i=0.0, max_i=1.0):
    mapa = gerar_ruido_rosa_2d(height, width, beta=2)
    return min_i + (max_i - min_i) * mapa

def blur_variavel(
    img_norm,
    min_sigma=0.5,
    max_sigma=3.0
):
    h, w, _ = img_norm.shape

    # Duas intensidades extremas
    sigma_fraco = min_sigma
    sigma_forte = max_sigma

    k1 = int(2 * np.ceil(3 * sigma_fraco) + 1)
    k2 = int(2 * np.ceil(3 * sigma_forte) + 1)

    blur_fraco = cv2.GaussianBlur(img_norm, (k1, k1), sigmaX=sigma_fraco)
    blur_forte = cv2.GaussianBlur(img_norm, (k2, k2), sigmaX=sigma_forte)

    # Mapa de mistura
    mapa = mapa_intensidade_blur(h, w, 0.0, 1.0)
    mapa = mapa[:, :, None]

    # Mistura contínua
    blurred = (1 - mapa) * blur_fraco + mapa * blur_forte

    return np.clip(blurred, 0, 1), mapa.mean()

def processar_dataset(
    input_dir,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

    labels = []

    for fname in tqdm(files, desc="Gerando ruído rosa"):
        path = os.path.join(input_dir, fname)

        img = Image.open(path).convert("RGB")
        img_np = np.asarray(img, dtype=np.float32) / 255.0

        noisy, intensidade_media = blur_variavel(
            img_np,
            min_sigma=0.8,
            max_sigma=4.0
        )

        noisy_uint8 = (noisy * 255).astype(np.uint8)
        out_img = Image.fromarray(noisy_uint8)

        out_path = os.path.join(output_dir, fname)
        out_img.save(
            out_path,
            quality=100,
            subsampling=0,
            optimize=False
        )

        labels.append((fname, intensidade_media))

    return labels


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    input_path = "/mnt/e/Datasets/NOISE_Aug/train/src"
    output_path = "/mnt/e/Datasets/NOISE_Aug/train/blur"

    labels = processar_dataset(
        input_path,
        output_path,
    )