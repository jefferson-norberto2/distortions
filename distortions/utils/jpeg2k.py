import os
import random
from PIL import Image
from tqdm import tqdm


def aplicar_jpeg2000_global(
    img,
    rate_range=(0.1, 1.0)
):
    rate = random.uniform(*rate_range)
    return img, rate


def processar_dataset_jpeg2000(
    input_dir,
    output_dir,
    rate_range=(0.1, 1.0)
):
    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    labels = []

    for fname in tqdm(files, desc="JPEG2000 global"):
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")

        img_out, rate = aplicar_jpeg2000_global(
            img,
            rate_range
        )

        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".jpeg")

        img_out.save(
            out_path,
            format="JPEG2000",
            quality_mode="rates",
            quality_layers=[rate]
        )

        labels.append((os.path.basename(out_path), rate))

    with open(os.path.join(output_dir, "labels_jpeg2000.txt"), "w") as f:
        for name, r in labels:
            f.write(f"{name},{r:.5f}\n")


if __name__ == "__main__":
    processar_dataset_jpeg2000(
        input_dir="/mnt/e/Datasets/NOISE_Aug/train/src",
        output_dir="/mnt/e/Datasets/NOISE_Aug/train/jpeg2k",
        rate_range=(0.08, 1.2)
    )
