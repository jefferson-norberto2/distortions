from pathlib import Path
import shutil

root = Path("/mnt/c/Users/jeff_/Pictures/database/train")  # caminho raiz do dataset

for class_dir in root.iterdir():
    if not class_dir.is_dir():
        continue

    target_dir = root / f"{class_dir.name}X"

    for img in class_dir.iterdir():
        if not img.is_file():
            continue

        # nome sem extensão termina com X?
        if img.stem.endswith("X"):
            target_dir.mkdir(exist_ok=True)
            shutil.move(str(img), target_dir / img.name)
