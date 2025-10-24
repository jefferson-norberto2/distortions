from PIL import Image
import os

def rotate_and_crop(file_name, input_folder, output_folder):
    file_path = os.path.join(input_folder, file_name)
    img = Image.open(file_path)
    width, height = img.size

    # Verifica se a imagem é 512x512
    if width != 512 or height != 512:
        raise ValueError(f"A imagem {file_name} deve ter 512x512 pixels")

    # Cria pasta de saída, se não existir
    os.makedirs(output_folder, exist_ok=True)

    # Lista de ângulos de rotação
    rotations = [45, -45]

    for angle in rotations:
        # Rotaciona a imagem (expand=True evita corte nas bordas)
        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)

        # Calcula recorte central 256x256
        rw, rh = rotated.size
        left = (rw - 256) // 2
        top = (rh - 256) // 2
        right = left + 256
        bottom = top + 256
        cropped = rotated.crop((left, top, right, bottom))

        # Gera o nome do arquivo de saída
        output_path = os.path.join(output_folder, f"rot{angle}_{file_name}.png")
        cropped.save(output_path)
        print(f"Salvo: {output_path}")

        rotated.close()
        cropped.close()

    img.close()


def rotate_images_in_directory(root_directory: str) -> None:
    # Cria a pasta principal de saída
    output_root = root_directory.rstrip('/').rstrip('\\') + '_rotation'
    os.makedirs(output_root, exist_ok=True)

    for folder_name in os.listdir(root_directory):
        input_folder = os.path.join(root_directory, folder_name)

        if os.path.isdir(input_folder):
            print(f"\nProcessando pasta: {folder_name}")

            # Cria pasta correspondente na estrutura de saída
            output_folder = os.path.join(output_root, folder_name)
            os.makedirs(output_folder, exist_ok=True)

            for file_name in os.listdir(input_folder):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    try:
                        rotate_and_crop(file_name, input_folder, output_folder)
                    except Exception as e:
                        print(f"Erro ao processar {file_name}: {e}")

if __name__ == "__main__":
    rotate_images_in_directory('/home/jmn/host/dev/Datasets/IQA/ECSIQ/')
