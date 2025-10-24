from PIL import Image
import os

def split_image(file_name, folder_path):
    file_path = os.path.join(folder_path, file_name)
    # Abre a imagem
    img = Image.open(file_path)
    width, height = img.size

    # Verifica se a imagem é 512x512
    if width != 512 or height != 512:
        raise ValueError(f"A imagem {file_name} deve ter 512x512 pixels")

    # Coordenadas dos recortes (x0, y0, x1, y1)
    boxes = [
        (0, 0, 256, 256),       # superior esquerda
        (256, 0, 512, 256),     # superior direita
        (0, 256, 256, 512),     # inferior esquerda
        (256, 256, 512, 512)    # inferior direita
    ]

    # Faz os recortes e salva
    for i, box in enumerate(boxes):
        part = img.crop(box)
        output_path = os.path.join(folder_path, f"part_{i+1}_{file_name}.png")
        part.save(output_path)
        print(f"Salvo: {output_path}")

    # Fecha o arquivo da imagem antes de remover
    img.close()

    # Remove o arquivo original
    try:
        os.remove(file_path)
        print(f"Removido arquivo original: {file_path}")
    except Exception as e:
        print(f"Erro ao remover {file_path}: {e}")

def split_images_in_directory(root_directory: str) -> None:
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)

        if os.path.isdir(folder_path):
            print(f"\nProcessando a pasta: {folder_name}")

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    try:
                        split_image(file_name, folder_path)
                    except Exception as e:
                        print(f"Erro ao processar {file_path}: {e}")

if __name__ == "__main__":
    split_images_in_directory('/home/jmn/host/dev/Datasets/IQA/ECSIQ/')
