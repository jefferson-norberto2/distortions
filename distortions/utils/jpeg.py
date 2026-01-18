import os
import random
from PIL import Image
from tqdm import tqdm
import numpy as np


def aplicar_jpeg_global(
    img,
    quality_range=(20, 90)
):
    quality = random.randint(*quality_range)
    return img, quality


def processar_dataset_jpeg(
    input_dir,
    output_dir,
    quality_range=(20, 90)
):
    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]


    for fname in tqdm(files, desc="JPEG global"):
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")

        img_out, quality = aplicar_jpeg_global(
            img,
            quality_range
        )

        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".jpeg")
        img_out.save(
            out_path,
            format="JPEG",
            quality=quality,
            subsampling=np.random.choice([1, 2, 3, 4, 5]),
            optimize=False
        )


if __name__ == "__main__":
    processar_dataset_jpeg(
        input_dir="/mnt/e/Datasets/NOISE_Aug/val/src",
        output_dir="/mnt/e/Datasets/NOISE_Aug/val/jpeg",
        quality_range=(2, 85)
    )
