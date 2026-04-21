import os
import yaml
from pathlib import Path

# Função atualizada conforme sua especificação (suportando MobileNet, YOLO e ResNet)
def get_model_parts(model_name: str):
    if '_' in model_name:
        parts = model_name.split('_', 1)
        return parts[0], parts[1].upper()
    else:
        family = 'yolo' if model_name.startswith('yolo') else 'resnet'
        parts = model_name.split(family, 1)
        version = parts[1] if len(parts) > 1 else 'UNKNOWN'
        return family, version

# Função para trocar o ponto pela vírgula (padrão PT-BR)
def format_ptbr(value):
    if value is None:
        return "N/A"
    return str(value).replace('.', ',')

def generate_csv_by_color():
    base_dir = Path('runs')
    
    # Adicionei alguns exemplos de resnet na lista para refletir sua nova função
    models = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x',
        'mobilenet_v1', 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    ]
    colors = ['RGB', 'LAB', 'HSV']

    header = "model;test_accuracy;cross_accuracy;System_RAM_Usage_GB_average;GPU_Processing_Usage_Percent_average;GPU_VRAM_Allocation_GB_average;Power_Consumption_Watts_average"
    
    # Vamos iterar primeiro pelas cores para criar um CSV para cada uma
    for color in colors:
        csv_lines = [header]
        
        for model_name in models:
            family, version = get_model_parts(model_name)
            model_dir = base_dir / 'tested' / family / version
            
            # --- 1. Acurácias ---
            test_acc = None
            test_dir = model_dir / 'test' / color / 'run_1'
            test_yaml = test_dir / 'informations.yaml'
            if test_yaml.exists():
                with open(test_yaml, 'r') as f:
                    info = yaml.safe_load(f)
                    if info:
                        test_acc = info.get('Accuracy_percent')

            cross_acc = None
            cross_dir = model_dir / 'cross_test' / color / 'run_1'
            cross_yaml = cross_dir / 'informations.yaml'
            if cross_yaml.exists():
                with open(cross_yaml, 'r') as f:
                    info = yaml.safe_load(f)
                    if info:
                        cross_acc = info.get('Accuracy_percent')

            # Se não existe dado de acurácia, o modelo não rodou para essa cor, então pulamos
            if test_acc is None and cross_acc is None:
                continue

            # --- 2. Hardware Específico ---
            # Vamos pegar o hardware da execução de teste (LIST). 
            # Se preferir a do cross_test, basta mudar para cross_dir
            ram_avg, gpu_avg, vram_avg, power_avg = None, None, None, None
            hw_file = test_dir / 'hardware_metrics.yaml'
            
            if hw_file.exists():
                with open(hw_file, 'r') as f:
                    hw_data = yaml.safe_load(f)
                    if hw_data:
                        ram_avg = hw_data.get('System_RAM_Usage_GB', {}).get('average')
                        gpu_avg = hw_data.get('GPU_Processing_Usage_Percent', {}).get('average')
                        vram_avg = hw_data.get('GPU_VRAM_Allocation_GB', {}).get('average')
                        power_avg = hw_data.get('Power_Consumption_Watts', {}).get('average')

            # --- 3. Montagem da Linha ---
            row = [
                model_name, # Aqui deixamos só o nome do modelo, pois a cor já estará no nome do arquivo
                format_ptbr(test_acc),
                format_ptbr(cross_acc),
                format_ptbr(ram_avg),
                format_ptbr(gpu_avg),
                format_ptbr(vram_avg),
                format_ptbr(power_avg)
            ]
            
            csv_lines.append(";".join(row))

        # --- 4. Salvando o CSV da Cor Atual ---
        # Só salva o arquivo se tiver mais que o cabeçalho
        if len(csv_lines) > 1:
            output_path = f'resultados_{color}.csv'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(csv_lines))
            print(f"CSV gerado com sucesso: {output_path}")

if __name__ == "__main__":
    generate_csv_by_color()