# --------------------------------------------------------------------------
# SCRIPT PARA FLIP VERTICAL E HORIZONTAL DE IMAGENS EM SUBPASTAS
# --------------------------------------------------------------------------

# Importa as bibliotecas necessárias:
# 'os' para interagir com o sistema de arquivos (pastas e arquivos).
# 'Image' da biblioteca PIL (Pillow) para manipular as imagens.
import os
from PIL import Image
from tqdm import tqdm

def flip_image(image_path: str, flip_type: str = 'horizontal') -> Image.Image:
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
        print(f"Erro ao processar a imagem {image_path}: {e}")
        return None


def flip_images_in_directory(directory: str = '/home/jmn/host/dev/Datasets/IQA/ECSIQ/') -> None:
    main_directory = directory

    print("🚀 Iniciando o script de flip de imagens...")

    for folder_name in tqdm(os.listdir(main_directory)):
        
        folder_path = os.path.join(main_directory, folder_name)

        if os.path.isdir(folder_path):
            print(f"\n📁 Processando a pasta: {folder_name}")

            for file_name in tqdm(os.listdir(folder_path)):
                
                file_path = os.path.join(folder_path, file_name)

                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    
                    horizontal_image = flip_image(file_path, flip_type='horizontal')
                    vertical_image = flip_image(file_path, flip_type='vertical')
                    both_image = flip_image(file_path, flip_type='both')

                    if horizontal_image:
                        horizontal_name = f"horizontal_{file_name}"
                        vertical_name = f"vertical_{file_name}"
                        both_name = f"both_{file_name}"
                        
                        horizontal_path = os.path.join(folder_path, horizontal_name)
                        vertical_path = os.path.join(folder_path, vertical_name)
                        both_path = os.path.join(folder_path, both_name)
                        
                        horizontal_image.save(horizontal_path)
                        vertical_image.save(vertical_path)
                        both_image.save(both_path)
            
if __name__ == "__main__":
    flip_images_in_directory('/home/jmn/host/dev/Datasets/IQA/ECSIQ/')