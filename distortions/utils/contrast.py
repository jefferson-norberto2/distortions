import os
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm


def aplicar_low_light_global(
    img,
    bright_range=(0.5, 0.9),
    contrast_range=(0.6, 0.95)
):
    bright_factor = random.uniform(*bright_range)
    contrast_factor = random.uniform(*contrast_range)

    img = ImageEnhance.Brightness(img).enhance(bright_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    return img


def processar_dataset(
    input_dir,
    output_dir,
    bright_range=(0.5, 0.9),
    contrast_range=(0.6, 0.95)
):
    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]


    for fname in tqdm(files, desc="Low-light (global)"):
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")

        img_out = aplicar_low_light_global(
            img, bright_range, contrast_range
        )

        img_out.save(
            os.path.join(output_dir, fname),
            quality=100,
            subsampling=0
        )



if __name__ == "__main__":
    processar_dataset(
        input_dir="/mnt/e/Datasets/NOISE_Aug/val/src",
        output_dir="/mnt/e/Datasets/NOISE_Aug/val/contrast",
        bright_range=(0.55, 0.85),
        contrast_range=(0.65, 0.9)
    )
