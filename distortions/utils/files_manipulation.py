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
                    folder_path: str, 
                    types: List[str]
                ) -> None:
        """
            Function to flip all images in a folder.

            parameters:
            - folder_path: Path to the folder containing images to be flipped.
            - types: List of flip types to apply ('horizontal', 'vertical', 'both')

        """
        print("🚀 Starting flip of images...")

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
        Crop a single image into multiple parts based on specified positions.

        parameters:
        - input_file_path: Path to the input image file.
        - output_path: Directory where cropped images will be saved.
        - crop_size: Size of the crop. Can be an integer (for square crops)
                     or a tuple (width, height).
        - positions: List of positions to crop from. Options include:
                     'top_left', 'top_right', 'bottom_left', 'bottom_right', 'center'.
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
            'top_center':   ((w_img - w_crop) // 2, 0, (w_img - w_crop) // 2 + w_crop, h_crop),
            'bottom_left':  (0, h_img - h_crop, w_crop, h_img),
            'bottom_right': (w_img - w_crop, h_img - h_crop, w_img, h_img),
            'bottom_center':( (w_img - w_crop) // 2, h_img - h_crop, (w_img - w_crop) // 2 + w_crop, h_img),
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
                'top_center', 
                'bottom_left', 
                'bottom_center',
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

    def merge_folders(self, 
                      source_folder: str, 
                      destination_folder: str
                  ) -> None:
        """
        Move all files from source_folder to destination_folder.
        If a file already exists in destination_folder, it will be skipped.
        """
        if source_folder == destination_folder:
            raise ValueError("Source and destination folders must be different.")

        if not os.path.exists(source_folder):
            raise FileNotFoundError(f"Source folder not found: {source_folder}")

        os.makedirs(destination_folder, exist_ok=True)

        for dirpath, dirnames, filenames in os.walk(source_folder):
            rel_path = os.path.relpath(dirpath, source_folder)
            dest_dir = os.path.join(destination_folder, rel_path)

            os.makedirs(dest_dir, exist_ok=True)

            for filename in filenames:
                src_file = os.path.join(dirpath, filename)
                dest_file = os.path.join(dest_dir, filename)

                if os.path.exists(dest_file):
                    print(f"⚠️  Skipping (already exists): {dest_file}")
                    continue

                try:
                    os.rename(src_file, dest_file)
                    print(f"✅ Moved: {dest_file}")
                except Exception as e:
                    print(f"Error moving {src_file}: {e}")

        print("\n✅ All files have been processed!")

if __name__ == "__main__":
    manipulator = FilesManipulation()
    
    base_dir = '/home/jmn/dev/Datasets/LIVE/'
    # folders = ['blur', 'awgn', 'jpeg', 'jpeg2000']
    
    # for folder in folders:
    #     manipulator.flip_images(
    #         folder_path=f"{base_dir}/{folder}",
    #         types=['horizontal', 'vertical', 'both']
    #     )
    
    folders = ['jpeg2000']
    
    for folder in folders:
        manipulator.crop_images(
            input_folder=f"{base_dir}/{folder}",
            output_folder=f"{base_dir}_512/{folder}",
            crop_size=512,
            positions=['center']
        )
