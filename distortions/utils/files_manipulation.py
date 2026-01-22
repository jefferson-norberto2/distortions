import os
from tqdm import tqdm
from PIL import Image
from typing import List, Tuple

class FilesManipulation:
    def __init__(self):
        pass

    def flip_single_image(self, 
                   image_path: str, 
                   flip_type: str
                ) -> Image.Image:
        """
        Função para flipar uma imagem.

        Parâmetros:
        - image_path: Caminho para a imagem a ser flipada.
        - flip_type: Tipo de flip ('horizontal' ou 'vertical').

        Retorna:
        - A imagem flipada.
        """
        # Abre a imagem usando PIL
        try:
            img = Image.open(image_path)

            # Verifica o tipo de flip e aplica
            if flip_type == 'horizontal':
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip_type == 'vertical':
                flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif flip_type == 'both':
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
            else:
                raise ValueError("flip_type deve ser 'horizontal' ou 'vertical'")

            return flipped_img
        except Exception as e:
            raise RuntimeError(f"Error to flip image {image_path}: {e}")

    def flip_images(self, 
                    directory: str, 
                    types: List[str]
                ) -> None:
        main_directory = directory

        print("🚀 Iniciando o script de flip de imagens...")

        for folder_name in tqdm(os.listdir(main_directory)):
            
            folder_path = os.path.join(main_directory, folder_name)

            if os.path.isdir(folder_path):
                print(f"\n📁 Processando a pasta: {folder_name}")

                for file_name in tqdm(os.listdir(folder_path)):
                    
                    file_path = os.path.join(folder_path, file_name)

                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        
                        for flip_type in types:

                            fliped_image = self.flip_single_image(file_path, flip_type=flip_type)

                            if fliped_image:
                                fliped_name = f"{flip_type}_{file_name}"
                                
                                fliped_path = os.path.join(folder_path, fliped_name)
                                
                                fliped_image.save(fliped_path)

    def crop_single_image(
            self, 
            input_file_path: str, 
            output_path: str, 
            crop_size: int | Tuple[int, int], 
            positions: List[str] = ['center']
        ):
        """
        Recorta a imagem em posições específicas baseadas no tamanho desejado.
        
        :param crop_size: Inteiro (para quadrados) ou tupla (w, h). Tamanho do recorte.
        :param positions: Lista com as posições desejadas. Opções:
                        'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right'.
                        Padrão é ['center'].
        """
        os.makedirs(output_path, exist_ok=True)
        
        file_name = os.path.basename(input_file_path)
        
        try:
            img = Image.open(input_file_path)
        except:
            raise FileNotFoundError(f"Image not found at path: {input_file_path}.")

        w_img, h_img = img.size
        
        if isinstance(crop_size, int):
            w_crop, h_crop = crop_size, crop_size
        else:
            w_crop, h_crop = crop_size

        if w_img < w_crop or h_img < h_crop:
            img.close()
            raise ValueError(f"Crop size {crop_size} is larger than image size {img.size} for file {file_name}.")

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

        for pos in positions:
            if pos in coords_map:
                box = coords_map[pos]
                part = img.crop(box)
                
                output_filename = f"{pos}_{file_name}" 
                
                output = os.path.join(output_path, output_filename)
                part.save(output)
            else:
                print(f"Warning: '{pos}' unknown. Skipping.")

        img.close()

    def crop_images(
            self, 
            input_folder: str, 
            output_folder: str,
            crop_size: int | Tuple[int, int], 
            positions=[
                'center', 
                'top_left', 
                'top_right', 
                'bottom_left', 
                'bottom_right'
                ]
            ) -> None:
        
        if input_folder == output_folder:
            raise ValueError("Input and output folders must be different.")

        os.makedirs(output_folder, exist_ok=True)

        for file_name in tqdm(os.listdir(input_folder), desc=f"Processing files in {input_folder}"):
            file_path = os.path.join(input_folder, file_name)

            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    self.crop_single_image(f'{input_folder}/{file_name}',
                                           output_folder, 
                                           crop_size=crop_size, 
                                           positions=positions)
                except Exception as e:
                    print(f"Warning: Was not able to process {file_path}: {e}")

if __name__ == "__main__":
    manipulator = FilesManipulation()
    manipulator.crop_single_image(
        input_file_path='/root/Documents/dev/python/distortions/data/LIVE/train/src_imgs/bikes.bmp',
        output_path='/root/Documents/dev/python/distortions/data/LIVE/train/src_imgs/',
        crop_size=256,
        positions=['center']
    )
