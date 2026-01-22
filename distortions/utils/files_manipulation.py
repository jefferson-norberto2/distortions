from PIL import Image
import os

class FilesManipulation:
    def __init__(self):
        pass

    def crop_single_image(self, file_name: str, folder_path: str):
        file_path = os.path.join(folder_path, file_name)
        img = Image.open(file_path)
        width, height = img.size

        if width != 512 or height != 512:
            raise ValueError(f"The {file_name} image need to be 512x512 pixels")

        boxes = [
            (0, 0, 256, 256),       # superior esquerda
            (256, 0, 512, 256),     # superior direita
            (0, 256, 256, 512),     # inferior esquerda
            (256, 256, 512, 512)    # inferior direita
        ]

        for i, box in enumerate(boxes):
            part = img.crop(box)
            output_path = os.path.join(folder_path, f"part_{i+1}_{file_name}.png")
            part.save(output_path)
            print(f"Salvo: {output_path}")

        img.close()

    def crop_images(self, root_directory: str) -> None:
        for folder_name in os.listdir(root_directory):
            folder_path = os.path.join(root_directory, folder_name)

            if os.path.isdir(folder_path):
                print(f"\nProcessando a pasta: {folder_name}")

                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        try:
                            self.crop_single_image(file_name, folder_path)
                        except Exception as e:
                            print(f"Erro ao processar {file_path}: {e}")

if __name__ == "__main__":
    manipulator = FilesManipulation()
    manipulator.crop_images('/home/jmn/host/dev/Datasets/IQA/ECSIQ/')
