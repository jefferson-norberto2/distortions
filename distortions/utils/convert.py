import cv2
from pathlib import Path
from tqdm import tqdm

def convert_dataset(src_root, target_root, color_space='HSV'):
    """
    src_root: Caminho da pasta original (ex: 'Datasets/Dist')
    target_root: Caminho da nova pasta (ex: 'Datasets/Dist_HSV')
    color_space: 'HSV' ou 'LAB'
    """
    src_path = Path(src_root)
    target_path = Path(target_root)
    
    # Mapeamento de conversão do OpenCV
    if color_space.upper() == 'HSV':
        conv_code = cv2.COLOR_BGR2HSV
    elif color_space.upper() == 'LAB':
        conv_code = cv2.COLOR_BGR2LAB
    else:
        print("Espaço de cor não suportado. Use 'HSV' ou 'LAB'.")
        return

    # Lista todos os arquivos de imagem recursivamente
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in src_path.rglob('*') if f.suffix.lower() in extensions]
    
    print(f"--- Iniciando conversão para {color_space} ---")
    print(f"Origem: {src_root} | Destino: {target_root}")

    for img_file in tqdm(image_files):
        # Define o caminho de destino mantendo a subpasta (classe)
        relative_path = img_file.relative_to(src_path)
        dest_file = target_path / relative_path
        
        # Cria as pastas de classe se não existirem
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Processamento
        img = cv2.imread(str(img_file))
        if img is not None:
            converted = cv2.cvtColor(img, conv_code)
            cv2.imwrite(str(dest_file), converted)
        else:
            print(f"Erro ao ler: {img_file}")

if __name__ == "__main__":
    # 1. Configura os caminhos (Ajuste conforme sua estrutura no Nitro V15)
    base_dir = "Datasets/LIVE_RGB"
    
    # 2. Executa as conversões
    convert_dataset(base_dir, f"LIVE_LAB", color_space='LAB')
    #convert_dataset(base_dir, f"{base_dir}_LAB", color_space='LAB')
    
    print("\nProcesso concluído! Agora você tem 3 datasets independentes.")