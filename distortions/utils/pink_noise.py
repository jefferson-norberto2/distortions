import os
import numpy as np
from PIL import Image
from tqdm import tqdm


# =========================================================
# Pink Noise 2D
# =========================================================
def gerar_ruido_rosa_2d(height, width, beta=2.0):
    freq_x = np.fft.fftfreq(width)
    freq_y = np.fft.fftfreq(height)
    fxx, fyy = np.meshgrid(freq_x, freq_y)

    f2 = fxx**2 + fyy**2
    f2[0, 0] = 1.0

    amplitude = 1.0 / (f2 ** (beta / 4.0))

    real = np.random.normal(0, 1, (height, width))
    imag = np.random.normal(0, 1, (height, width))
    ruido_fft = real + 1j * imag

    filtrado = ruido_fft * amplitude
    ruido = np.real(np.fft.ifft2(filtrado))

    ruido = (ruido - ruido.min()) / (ruido.max() - ruido.min())
    return ruido


# =========================================================
# Intensity Maps
# =========================================================
def mapa_intensidade_rosa(height, width, min_i=0.05, max_i=0.3):
    mapa = gerar_ruido_rosa_2d(height, width, beta=2)
    return min_i + (max_i - min_i) * mapa


def mapa_intensidade_luminancia(img_norm, min_i=0.05, max_i=0.3):
    luma = (
        0.299 * img_norm[:, :, 0] +
        0.587 * img_norm[:, :, 1] +
        0.114 * img_norm[:, :, 2]
    )

    luma_inv = 1.0 - luma
    ruido_suave = gerar_ruido_rosa_2d(*luma.shape, beta=2)

    mapa = luma_inv * ruido_suave
    mapa = (mapa - mapa.min()) / (mapa.max() - mapa.min())

    return min_i + (max_i - min_i) * mapa


# =========================================================
# Noise Application
# =========================================================
def aplicar_ruido_rosa(
    img_norm,
    modo="luminancia",  # "rosa" | "luminancia"
    min_i=0.05,
    max_i=0.3
):
    h, w, _ = img_norm.shape

    ruido_rgb = np.stack([
        gerar_ruido_rosa_2d(h, w),
        gerar_ruido_rosa_2d(h, w),
        gerar_ruido_rosa_2d(h, w)
    ], axis=-1)

    if modo == "rosa":
        intensidade = mapa_intensidade_rosa(h, w, min_i, max_i)
    else:
        intensidade = mapa_intensidade_luminancia(img_norm, min_i, max_i)

    intensidade_rgb = intensidade[:, :, None]

    noisy = img_norm + intensidade_rgb * ruido_rgb
    noisy = np.clip(noisy, 0, 1)

    return noisy, intensidade.mean()


# =========================================================
# Dataset Processing
# =========================================================
def processar_dataset(
    input_dir,
    output_dir,
    modo_ruido="luminancia"
):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

    labels = []

    for fname in tqdm(files, desc="Gerando ruído rosa"):
        path = os.path.join(input_dir, fname)

        img = Image.open(path).convert("RGB")
        img_np = np.asarray(img, dtype=np.float32) / 255.0

        noisy, intensidade_media = aplicar_ruido_rosa(
            img_np,
            modo=modo_ruido
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
    output_path = "/mnt/e/Datasets/NOISE_Aug/train/pink_noise"

    labels = processar_dataset(
        input_path,
        output_path,
        modo_ruido="luminancia"  # ou "rosa"
    )
