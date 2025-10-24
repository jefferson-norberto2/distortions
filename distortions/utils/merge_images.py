import os
import shutil

def merge_rotated_images(original_root: str):
    rotation_root = original_root.rstrip('/').rstrip('\\') + '_rotation'

    if not os.path.exists(rotation_root):
        print(f"❌ A pasta {rotation_root} não existe.")
        return

    print(f"🔄 Movendo arquivos de '{rotation_root}' → '{original_root}'")

    # Percorre toda a estrutura da pasta de rotações
    for dirpath, dirnames, filenames in os.walk(rotation_root):
        # Caminho relativo (mantém estrutura)
        rel_path = os.path.relpath(dirpath, rotation_root)
        dest_dir = os.path.join(original_root, rel_path)

        # Garante que a pasta de destino exista
        os.makedirs(dest_dir, exist_ok=True)

        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            dest_file = os.path.join(dest_dir, filename)

            # Se já existir, pula
            if os.path.exists(dest_file):
                print(f"⚠️  Pulando (já existe): {dest_file}")
                continue

            # Move o arquivo
            try:
                shutil.move(src_file, dest_file)
                print(f"✅ Movido: {dest_file}")
            except Exception as e:
                print(f"Erro ao mover {src_file}: {e}")

    print("\n✅ Todos os arquivos foram processados!")

if __name__ == "__main__":
    merge_rotated_images('/home/jmn/host/dev/Datasets/IQA/ECSIQ/')
