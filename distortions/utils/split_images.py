from PIL import Image
import os
from typing import List

def split_image(file_name: str, folder_path: str, output_folder: str, crop_size=299, positions: List[str] | None =None):
    """
    Recorta a imagem em posições específicas baseadas no tamanho desejado.
    
    :param crop_size: Inteiro (para quadrados) ou tupla (w, h). Tamanho do recorte.
    :param positions: Lista com as posições desejadas. Opções:
                      'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right'.
                      Se None, padrão é ['center'].
    """
    if positions is None:
        positions = ['center']
        
    file_path = os.path.join(folder_path, file_name)
    
    # Abre a imagem
    try:
        img = Image.open(file_path)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {file_path}")
        return

    w_img, h_img = img.size
    
    # Tratamento se crop_size for apenas um inteiro (quadrado)
    if isinstance(crop_size, int):
        w_crop, h_crop = crop_size, crop_size
    else:
        w_crop, h_crop = crop_size

    # Verificação básica
    if w_img < w_crop or h_img < h_crop:
        print(f"Erro: A imagem {file_name} ({w_img}x{h_img}) é menor que o recorte ({w_crop}x{h_crop}).")
        img.close()
        return

    # Dicionário com a lógica de cálculo para cada posição
    # As coordenadas são (left, top, right, bottom)
    coords_map = {
        'top_left':     (0, 0, w_crop, h_crop),
        'top_right':    (w_img - w_crop, 0, w_img, h_crop),
        'bottom_left':  (0, h_img - h_crop, w_crop, h_img),
        'bottom_right': (w_img - w_crop, h_img - h_crop, w_img, h_img),
        'center':       (
            (w_img - w_crop) // 2,
            (h_img - h_crop) // 2,
            (w_img - w_crop) // 2 + w_crop,
            (h_img - h_crop) // 2 + h_crop
        )
    }

    processed = False

    for pos in positions:
        if pos in coords_map:
            box = coords_map[pos]
            part = img.crop(box)
            
            # Nome do arquivo ajustado para indicar a posição
            output_filename = f"{pos}_{file_name}" 
            # Se preferir manter o padrão 'part_X', você pode usar um contador
            
            output_path = os.path.join(output_folder, output_filename)
            part.save(output_path)
            print(f"Salvo: {output_path}")
            processed = True
        else:
            print(f"Aviso: Posição '{pos}' desconhecida. Pulando.")

    img.close()

def split_images_in_directory(root_directory: str, crop_size=256, positions=['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']) -> None:
    output_directory = f'{root_directory}_croped'
    os.makedirs(output_directory, exist_ok=True)
    
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        
        new_folder_path = os.path.join(output_directory, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        if os.path.isdir(folder_path):
            print(f"\nProcessando a pasta: {folder_name}")

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    try:
                        split_image(file_name, folder_path, new_folder_path, crop_size=crop_size, positions=positions)
                    except Exception as e:
                        print(f"Erro ao processar {file_path}: {e}")

if __name__ == "__main__":
    split_images_in_directory(root_directory='data/MYCSIQ/val', crop_size=299)
