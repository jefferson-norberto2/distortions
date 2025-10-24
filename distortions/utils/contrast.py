import os
import random
import sys
from PIL import Image, ImageEnhance, UnidentifiedImageError

# Define as extensões de imagem permitidas
ALLOWED_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png'}

def get_float_input(prompt):
    """Pede ao usuário um número float e valida a entrada."""
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Entrada inválida. Por favor, insira um número (ex: 0.8).")

def process_directory(directory, bright_range, contrast_range):
    """
    Processa todos os arquivos no diretório.
    Remove arquivos não-imagem e ajusta imagens válidas.
    """
    print(f"\nIniciando processamento no diretório: {directory}")
    
    # Desempacota os ranges
    min_bright, max_bright = bright_range
    min_contrast, max_contrast = contrast_range

    if not os.path.isdir(directory):
        print(f"Erro: O diretório '{directory}' não foi encontrado.")
        return

    # Lista todos os itens no diretório
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ignora se for um subdiretório
        if not os.path.isfile(file_path):
            print(f"Ignorando (é um diretório): {filename}")
            continue

        # Pega a extensão do arquivo em minúsculas
        _, ext = os.path.splitext(filename)
        ext_lower = ext.lower()

        if ext_lower in ALLOWED_EXTENSIONS:
            # --- É UMA IMAGEM: PROCESSAR ---
            try:
                # Abre a imagem
                with Image.open(file_path) as img:
                    
                    # 1. Gera fatores aleatórios
                    bright_factor = random.uniform(min_bright, max_bright)
                    contrast_factor = random.uniform(min_contrast, max_contrast)

                    # 2. Aplica brilho
                    enhancer_brightness = ImageEnhance.Brightness(img)
                    img_adjusted = enhancer_brightness.enhance(bright_factor)

                    # 3. Aplica contraste
                    enhancer_contrast = ImageEnhance.Contrast(img_adjusted)
                    img_final = enhancer_contrast.enhance(contrast_factor)
                    
                    # 4. Salva a imagem (sobrescreve a original)
                    img_final.save(file_path)

                    print(f"Processado: {filename} (Brilho: {bright_factor:.2f}, Contraste: {contrast_factor:.2f})")

            except UnidentifiedImageError:
                print(f"Erro: Não foi possível identificar a imagem: {filename}. Pode estar corrompida.")
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
        
        else:
            # --- NÃO É UMA IMAGEM: REMOVER ---
            try:
                os.remove(file_path)
                print(f"Removido (não é imagem): {filename}")
            except PermissionError:
                print(f"Erro de permissão: Não foi possível remover {filename}.")
            except Exception as e:
                print(f"Erro ao remover {filename}: {e}")

def main():
    print("--- Modificador Aleatório de Brilho e Contraste ---")
    print("\nAVISO: Este script MODIFICA PERMANENTEMENTE as imagens originais")
    print("e REMOVE PERMANENTEMENTE arquivos que não são imagens válidas")
    print(f"(permitidos: {', '.join(ALLOWED_EXTENSIONS)}) no diretório de destino.")
    print("\n*** FAÇA UM BACKUP ANTES DE CONTINUAR! ***\n")

    # 1. Obter o diretório
    directory_path = input("Insira o caminho completo para o diretório: ").strip()

    # 2. Obter os ranges
    print("\nInsira os ranges para os fatores de ajuste:")
    print("(1.0 = original, 0.7 = 70% de redução, 1.2 = 20% de aumento)")
    
    min_bright = get_float_input("Brilho MÍNIMO (ex: 0.6): ")
    max_bright = get_float_input("Brilho MÁXIMO (ex: 0.9): ")
    min_contrast = get_float_input("Contraste MÍNIMO (ex: 0.7): ")
    max_contrast = get_float_input("Contraste MÁXIMO (ex: 1.0): ")

    # Validação simples dos ranges
    if min_bright > max_bright or min_contrast > max_contrast:
        print("\nErro: O valor MÍNIMO não pode ser maior que o valor MÁXIMO.")
        sys.exit(1) # Encerra o script

    # 3. Confirmação final
    print("\nResumo da operação:")
    print(f"Diretório: {directory_path}")
    print(f"Range Brilho: [{min_bright}, {max_bright}]")
    print(f"Range Contraste: [{min_contrast}, {max_contrast}]")
    print("Arquivos não-imagem SERÃO REMOVIDOS.")
    
    confirm = input("\nVocê tem certeza que deseja continuar? (s/n): ").strip().lower()

    if confirm == 's':
        process_directory(directory_path, (min_bright, max_bright), (min_contrast, max_contrast))
        print("\nProcessamento concluído.")
    else:
        print("Operação cancelada.")

if __name__ == "__main__":
    main()