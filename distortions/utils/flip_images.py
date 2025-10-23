# --------------------------------------------------------------------------
# SCRIPT PARA FLIP VERTICAL E HORIZONTAL DE IMAGENS EM SUBPASTAS
# --------------------------------------------------------------------------

# Importa as bibliotecas necessárias:
# 'os' para interagir com o sistema de arquivos (pastas e arquivos).
# 'Image' da biblioteca PIL (Pillow) para manipular as imagens.
import os
from PIL import Image
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
# Define o diretório principal. O '.' significa "o diretório atual",
# ou seja, a pasta onde este script Python está salvo.
diretorio_principal = '/home/jmn/dev/Datasets/IQA/ECSIQ/'

# --- EXECUÇÃO DO SCRIPT ---
print("🚀 Iniciando o script de flip de imagens...")

# 1. Percorrer cada item (arquivo ou pasta) no diretório principal.
for nome_da_pasta in tqdm(os.listdir(diretorio_principal)):
    
    # Constrói o caminho completo para a pasta.
    # Ex: se nome_da_pasta for "blur", caminho_da_pasta será "./blur"
    caminho_da_pasta = os.path.join(diretorio_principal, nome_da_pasta)

    # 2. Verifica se o item atual é realmente uma pasta (diretório).
    if os.path.isdir(caminho_da_pasta):
        print(f"\n📁 Processando a pasta: {nome_da_pasta}")

        # 3. Percorre cada arquivo dentro da subpasta.
        for nome_do_arquivo in tqdm(os.listdir(caminho_da_pasta)):
            
            # Constrói o caminho completo para o arquivo de imagem.
            caminho_do_arquivo = os.path.join(caminho_da_pasta, nome_do_arquivo)

            # Usamos um bloco 'try...except' para garantir que, se o arquivo
            # não for uma imagem, o script não pare e continue para o próximo.
            try:
                # 4. Abre a imagem original. O 'with' garante que o arquivo seja fechado corretamente.
                with Image.open(caminho_do_arquivo) as img_original:
                    # print(f"  -> Processando imagem: {nome_do_arquivo}")

                    # Separa o nome base do arquivo e sua extensão.
                    # Ex: "foto.jpg" vira ("foto", ".jpg")
                    nome_base, extensao = os.path.splitext(nome_do_arquivo)

                    # --- FLIP VERTICAL ---
                    # 5. Cria uma nova imagem com o flip vertical a partir da ORIGINAL.
                    img_vertical = img_original.transpose(Image.FLIP_TOP_BOTTOM)
                    
                    # 6. Define o novo nome para a imagem com flip vertical.
                    novo_nome_vertical = f"{nome_base}_vertical{extensao}"
                    caminho_salvar_vertical = os.path.join(caminho_da_pasta, novo_nome_vertical)
                    
                    # 7. Salva a nova imagem.
                    img_vertical.save(caminho_salvar_vertical)
                    # print(f"     ✅ Salvo como: {novo_nome_vertical}")

                    # --- FLIP HORIZONTAL ---
                    # 8. Cria uma nova imagem com o flip horizontal a partir da ORIGINAL.
                    img_horizontal = img_original.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    # 9. Define o novo nome para a imagem com flip horizontal.
                    novo_nome_horizontal = f"{nome_base}_horizontal{extensao}"
                    caminho_salvar_horizontal = os.path.join(caminho_da_pasta, novo_nome_horizontal)

                    # 10. Salva a nova imagem.
                    img_horizontal.save(caminho_salvar_horizontal)
                    # print(f"     ✅ Salvo como: {novo_nome_horizontal}")

                    # --- FLIP VERTICAL + HORIZONTAL ---
                    # 8. Cria uma nova imagem com o flip horizontal a partir da ORIGINAL.
                    img_both = img_vertical.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    # 9. Define o novo nome para a imagem com flip horizontal.
                    novo_nome_both = f"{nome_base}_both{extensao}"
                    caminho_salvar_both = os.path.join(caminho_da_pasta, novo_nome_both)

                    # 10. Salva a nova imagem.
                    img_horizontal.save(caminho_salvar_both)
                    # print(f"     ✅ Salvo como: {novo_nome_both}")

            except IOError:
                # Se o arquivo não for um formato de imagem válido, avisa e ignora.
                print(f"  -> ⚠️ O arquivo '{nome_do_arquivo}' não parece ser uma imagem. Ignorando.")

print("\n🎉 Processo concluído com sucesso!")