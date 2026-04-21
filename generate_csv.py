import os
import yaml
from pathlib import Path

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
    # Converte para string e troca o ponto por vírgula
    return str(value).replace('.', ',')

def generate_csv():
    base_dir = Path('runs')
    
    # Modelos e cores que você utilizou nos testes
    models = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x',
        'mobilenet_v1', 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    ]
    colors = ['RGB', 'LAB', 'HSV']

    # Cabeçalho exato solicitado (corrigi apenas 'avarage' para 'average')
    header = "model;test_accuracy;cross_accuracy;System_RAM_Usage_GB_average;GPU_Processing_Usage_Percent_average;GPU_VRAM_Allocation_GB_average;Power_Consumption_Watts_average"
    
    csv_lines = [header]

    for model_name in models:
        family, version = get_model_parts(model_name)
        model_dir = base_dir / family / version
        
        # 1. Carrega os dados de Hardware Globais do modelo
        hw_file = model_dir / 'global_hardware_metrics.yaml'
        hw_data = {}
        if hw_file.exists():
            with open(hw_file, 'r') as f:
                data = yaml.safe_load(f)
                if data and model_name in data:
                    hw_data = data[model_name]
        
        # Extrai as médias de hardware (retorna None se não achar)
        ram_avg = hw_data.get('System_RAM_Usage_GB', {}).get('average')
        gpu_avg = hw_data.get('GPU_Processing_Usage_Percent', {}).get('average')
        vram_avg = hw_data.get('GPU_VRAM_Allocation_GB', {}).get('average')
        power_avg = hw_data.get('Power_Consumption_Watts', {}).get('average')

        # 2. Varre as cores para pegar as acurácias individuais
        for color in colors:
            row_model_name = f"{model_name}_{color}"
            
            # Pega Acurácia do Teste Normal (LIST)
            test_acc = None
            test_yaml = model_dir / 'test' / color / 'run_1' / 'informations.yaml'
            if test_yaml.exists():
                with open(test_yaml, 'r') as f:
                    info = yaml.safe_load(f)
                    if info:
                        test_acc = info.get('Accuracy_percent')

            # Pega Acurácia do Cross Test (CSIQ)
            cross_acc = None
            cross_yaml = model_dir / 'cross_test' / color / 'run_1' / 'informations.yaml'
            if cross_yaml.exists():
                with open(cross_yaml, 'r') as f:
                    info = yaml.safe_load(f)
                    if info:
                        cross_acc = info.get('Accuracy_percent')

            # Se esse teste específico não rodou ou falhou, pulamos para não sujar o CSV
            # Se quiser que ele apareça mesmo assim com "N/A", basta comentar as duas linhas abaixo
            if test_acc is None and cross_acc is None:
                continue

            # Monta a linha do CSV aplicando a formatação pt-br
            row = [
                row_model_name,
                format_ptbr(test_acc),
                format_ptbr(cross_acc),
                format_ptbr(ram_avg),
                format_ptbr(gpu_avg),
                format_ptbr(vram_avg),
                format_ptbr(power_avg)
            ]
            
            csv_lines.append(";".join(row))

    # Salva o arquivo final
    output_path = 'resultados_finais.csv'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(csv_lines))
        
    print(f"CSV gerado com sucesso em: {output_path}")

if __name__ == "__main__":
    generate_csv()