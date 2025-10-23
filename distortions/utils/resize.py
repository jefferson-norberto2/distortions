import argparse
from pathlib import Path
from PIL import Image, ImageOps

def center_crop_to_min(img: Image.Image) -> Image.Image:
    # aplica orientação EXIF e faz crop central pelo menor lado
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return img.crop((left, top, left + m, top + m))

def process_file(path: Path, out_dir: Path, overwrite: bool):
    try:
        with Image.open(path) as img:
            cropped = center_crop_to_min(img)
            out_dir.mkdir(parents=True, exist_ok=True)
            if overwrite:
                save_path = path
            else:
                stem = path.stem + "_center"
                save_path = out_dir / (stem + path.suffix)
            # manter qualidade/format se possível
            save_params = {}
            if cropped.mode in ("RGBA", "P") and path.suffix.lower() in (".jpg", ".jpeg"):
                cropped = cropped.convert("RGB")
            cropped.save(save_path, **save_params)
            print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Failed {path}: {e}")

def find_images_in_dir(dirpath: Path, exts, recursive: bool):
    patterns = [f"**/*{e}" if recursive else f"*{e}" for e in exts]
    for p in patterns:
        yield from dirpath.glob(p)

def main():
    input_dir = './images'
    output_dir = './output'

    inp = Path(input_dir)
    out_dir = Path(output_dir)
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    if inp.is_file():
        process_file(inp, out_dir, False)
    elif inp.is_dir():
        for path in find_images_in_dir(inp, exts, True):
            if path.is_file():
                process_file(path, out_dir, False)
    else:
        print("Entrada não encontrada:", inp)

if __name__ == "__main__":
    main()