from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np

def mirror_pad_numpy(img, target_size=512):
    arr = np.array(img)
    h, w = arr.shape[:2]

    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="reflect"
    )

    return Image.fromarray(padded)


def center_crop_to_min(img: Image.Image) -> Image.Image:
    # aplica orientação EXIF e faz crop central pelo menor lado
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return img.crop((left, top, left + m, top + m))

def square_center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    img = center_crop_to_min(img)
    return img.resize((size, size), Image.LANCZOS)

def process_file(path: Path, out_dir: Path, overwrite: bool):
    try:
        with Image.open(path) as img:
            cropped = square_center_crop_resize(img, 512)
            out_dir.mkdir(parents=True, exist_ok=True)
            if overwrite:
                save_path = path
            else:
                stem = path.stem + "_center"
                save_path = out_dir / (stem + path.suffix)
            save_params = {
                "quality": 100,
                "subsampling": 0,
                "optimize": False
            }
            if cropped.mode in ("RGBA", "P") and path.suffix.lower() in (".jpg", ".jpeg"):
                cropped = cropped.convert("RGB")
            cropped.save(save_path.with_suffix(".png"), **save_params)
    except Exception as e:
        print(f"Failed {path}: {e}")

def find_images_in_dir(dirpath: Path, exts, recursive: bool):
    patterns = [f"**/*{e}" if recursive else f"*{e}" for e in exts]
    for p in patterns:
        yield from dirpath.glob(p)

def main_crop():
    input_dir = '/mnt/e/Datasets/UHD-IQA/src'
    output_dir = '/mnt/e/Datasets/UHD-IQA/src_cropped_4'

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    if input_path.is_file():
        process_file(input_path, output_path, False)
    elif input_path.is_dir():
        for path in tqdm(find_images_in_dir(input_path, exts, True)):
            if path.is_file():
                process_file(path, output_path, False)
    else:
        print("Entrada não encontrada:", input_path)

def main_pad():
    input_dir = '/mnt/e/Datasets/KADID10K/src'
    output_dir = '/mnt/e/Datasets/KADID10K/src_padded'

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    if input_path.is_file():
        mirror_pad_numpy(input_path, output_path, False)
    elif input_path.is_dir():
        for path in tqdm(find_images_in_dir(input_path, exts, True)):
            if path.is_file():
                img = Image.open(path)
                pad = mirror_pad_numpy(img)
                rel_path = path.relative_to(input_path)
                save_path = output_path / rel_path
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_options = {
                    "quality": 100,
                    "subsampling": 0,
                    "optimize": False
                }
                if pad.mode in ("RGBA", "P") and path.suffix.lower() in (".jpg", ".jpeg"):
                    pad = pad.convert("RGB")
                pad.save(save_path, **save_options)
    else:
        print("Entrada não encontrada:", input_path)

if __name__ == "__main__":
    main_crop()
    #main_pad()
    pass