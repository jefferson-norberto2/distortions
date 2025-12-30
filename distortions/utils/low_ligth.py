import os
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm


# =========================================================
# Utils
# =========================================================
def gerar_mapa_suave(h, w, min_v, max_v):
    mapa = np.random.rand(h, w)
    mapa = (mapa - mapa.min()) / (mapa.max() - mapa.min())
    return min_v + (max_v - min_v) * mapa


# =========================================================
# Aplicar brilho e contraste variável
# =========================================================
def aplicar_brilho_contraste_variavel(
    img_norm,
    bright_range=(0.7, 1.1),
    contrast_range=(0.7, 1.1)
):
    h, w, _ = img_norm.shape

    # Mapas de intensidade
    bright_map = gerar_mapa_suave(h, w, *bright_range)
    contrast_map = gerar_mapa_suave(h, w, *contrast_range)

    # Converter para PIL (necessário para ImageEnhance)
    img_pil = Image.fromarray((img_norm * 255).astype(np.uint8))

    # Aplicação GLOBAL base
    bright_base = np.mean(bright_map)
    contrast_base = np.mean(contrast_map)

    img_bc = ImageEnhance.Brightness(img_pil).enhance(bright_base)
    img_bc = ImageEnhance.Contrast(img_bc).enhance(contrast_base)

    img_bc_np = np.asarray(img_bc, dtype=np.float32) / 255.0

    # Refinamento espacial (mistura)
    alpha = (bright_map - bright_range[0]) / (bright_range[1] - bright_range[0])
    alpha = alpha[:, :, None]

    img_final = (1 - alpha) * img_norm + alpha * img_bc_np
    img_final = np.clip(img_final, 0, 1)

    label = {
        "brightness": bright_base,
        "contrast": contrast_base
    }

    return img_final, label


# =========================================================
# Dataset processing
# =========================================================
def processar_dataset(
    input_dir,
    output_dir,
    bright_range=(0.7, 1.1),
    contrast_range=(0.7, 1.1)
):
    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    labels = []

    for fname in tqdm(files, desc="Brilho & Contraste"):
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        img_np = np.asarray(img, dtype=np.float32) / 255.0

        out, label = aplicar_brilho_contraste_variavel(
            img_np, bright_range, contrast_range
        )

        out_uint8 = (out * 255).astype(np.uint8)
        Image.fromarray(out_uint8).save(
            os.path.join(output_dir, fname),
            quality=100,
            subsampling=0
        )

        labels.append((fname, label["brightness"], label["contrast"]))

    # Salvar labels
    with open(os.path.join(output_dir, "labels_bc.txt"), "w") as f:
        for name, b, c in labels:
            f.write(f"{name},{b:.4f},{c:.4f}\n")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    input_path = "/mnt/e/Datasets/NOISE_Aug/train/src"
    output_path = "/mnt/e/Datasets/NOISE_Aug/train/contrast"

    processar_dataset(
        input_path,
        output_path,
        bright_range=(0.3, 1.1),
        contrast_range=(0.5, 1.2)
    )
