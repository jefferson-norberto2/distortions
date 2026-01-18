import os
from PIL import Image

# =================================================

def garantir_pasta(caminho):
    """Verifica se a pasta existe, se não, cria."""
    if not os.path.exists(caminho):
        os.makedirs(caminho)
        print(f"Pasta criada: {caminho}")

def aplicar_ruido_compressoes():
    # Define os caminhos finais de saída
    pasta_jpeg = os.path.join(PASTA_DESTINO_BASE, "jpeg2")
    pasta_jpeg2k = os.path.join(PASTA_DESTINO_BASE, "jpeg2k2")

    # Garante que as pastas de origem e destino existam
    if not os.path.exists(PASTA_ORIGEM):
        print(f"Erro: A pasta de origem '{PASTA_ORIGEM}' não existe.")
        print("Por favor, crie a pasta e coloque suas imagens nela.")
        return

    garantir_pasta(pasta_jpeg)
    garantir_pasta(pasta_jpeg2k)

    arquivos = os.listdir(PASTA_ORIGEM)
    contador = 0

    print(f"Iniciando processamento de imagens em: {PASTA_ORIGEM}...\n")

    for arquivo in arquivos:
        if arquivo.lower().endswith(EXTENSOES_VALIDAS):
            caminho_completo_origem = os.path.join(PASTA_ORIGEM, arquivo)
            nome_sem_extensao = os.path.splitext(arquivo)[0]

            try:
                with Image.open(caminho_completo_origem) as img:
                    # Converter para RGB caso seja RGBA (PNG transparente),
                    # pois JPEG não suporta transparência.
                    if img.mode in ('RGBA', 'P'):
                         img = img.convert('RGB')

                    # --- Processar JPEG ---
                    caminho_destino_jpeg = os.path.join(pasta_jpeg, nome_sem_extensao + ".jpg")
                    # Salva com qualidade baixa para gerar os artefatos de bloco
                    img.save(caminho_destino_jpeg, 'JPEG', quality=QUALIDADE_JPEG)
                    print(f"[JPEG] Salvo: {caminho_destino_jpeg} (Qualidade: {QUALIDADE_JPEG})")

                    # --- Processar JPEG 2000 ---
                    caminho_destino_jp2 = os.path.join(pasta_jpeg2k, nome_sem_extensao + ".jp2")
                    # Para gerar ruído visível no JP2, precisamos usar compressão irreversível (lossy).
                    # Definimos a taxa de compressão alvo.
                    img.save(caminho_destino_jp2, 'JPEG2000', 
                             quality_mode='rates',
                             quality_layers=[TAXA_COMPRESSAO_JP2],
                             irreversible=True)
                    print(f"[JP2K] Salvo: {caminho_destino_jp2} (Taxa: {TAXA_COMPRESSAO_JP2}:1)")

                    contador += 1
                    print("-" * 30)

            except Exception as e:
                print(f"Erro ao processar {arquivo}: {e}")

    print(f"\nConcluído! {contador} imagens processadas.")
    print(f"Imagens JPEG salvas em: {pasta_jpeg}")
    print(f"Imagens JPEG 2000 salvas em: {pasta_jpeg2k}")

# ================= CONFIGURAÇÕES =================

# Pasta onde estão suas imagens originais
PASTA_ORIGEM = "/mnt/e/Datasets/NOISE_Aug/train/src"

# Pasta base onde as pastas de saída serão criadas
PASTA_DESTINO_BASE = "/mnt/e/Datasets/NOISE_Aug/train/"

# Extensões de imagem que o script vai procurar
EXTENSOES_VALIDAS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

# --- Configurações de Ruído JPEG ---
# Qualidade varia de 1 (pior qualidade, mais ruído) a 95 (melhor qualidade).
# Valores abaixo de 20 geram ruído de bloco bem visível.
QUALIDADE_JPEG = 10

# --- Configurações de Ruído JPEG 2000 (JP2) ---
# Para JP2, a compressão funciona de forma diferente.
# Usamos camadas de qualidade baseadas em taxa de compressão.
# Quanto MAIOR a taxa, MAIOR a compressão e o ruído.
# Exemplo: 100 significa compressão de 100:1.
TAXA_COMPRESSAO_JP2 = 80

if __name__ == "__main__":
    # Cria a pasta de origem se ela não existir, para facilitar o teste
    garantir_pasta(PASTA_ORIGEM)
    
    print("--- Gerador de Ruído de Compressão ---")
    print(f"Coloque suas imagens originais na pasta: '{PASTA_ORIGEM}'")
    input("Pressione Enter quando estiver pronto para começar...")
    
    aplicar_ruido_compressoes()