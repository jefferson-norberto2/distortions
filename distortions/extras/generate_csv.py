import yaml
from pathlib import Path
from distortions.utils.functions import extract_model_parts

def format_ptbr(value):
    if value is None:
        return "N/A"
    return str(value).replace('.', ',')

def generate_hardware_csv():
    # Diretório base atualizado conforme solicitado
    base_dir = Path('runs/tested')
    
    models = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x',
        'mobilenet_V1', 'mobilenet_V2', 'mobilenet_V3_small', 'mobilenet_V3_large',
    ]

    # Cabeçalho expandido para incluir as métricas de pico (peak)
    header = (
        "model;"
        "System_RAM_Usage_GB_average;System_RAM_Usage_GB_peak;"
        "GPU_Processing_Usage_Percent_average;GPU_Processing_Usage_Percent_peak;"
        "GPU_VRAM_Allocation_GB_average;GPU_VRAM_Allocation_GB_peak;"
        "Power_Consumption_Watts_average;Power_Consumption_Watts_peak"
    )
    
    csv_lines = [header]

    for model_name in models:
        family, version = extract_model_parts(model_name)
        model_dir = base_dir / family / version
        
        hw_file = model_dir / 'global_hardware_metrics.yaml'
        hw_data = {}
        
        if hw_file.exists():
            with open(hw_file, 'r') as f:
                data = yaml.safe_load(f)
                if data and model_name in data:
                    hw_data = data[model_name]
        
        # Extração de Médias (average)
        ram_avg = hw_data.get('System_RAM_Usage_GB', {}).get('average')
        gpu_avg = hw_data.get('GPU_Processing_Usage_Percent', {}).get('average')
        vram_avg = hw_data.get('GPU_VRAM_Allocation_GB', {}).get('average')
        power_avg = hw_data.get('Power_Consumption_Watts', {}).get('average')

        # Extração de Picos (peak)
        ram_peak = hw_data.get('System_RAM_Usage_GB', {}).get('peak')
        gpu_peak = hw_data.get('GPU_Processing_Usage_Percent', {}).get('peak')
        vram_peak = hw_data.get('GPU_VRAM_Allocation_GB', {}).get('peak')
        power_peak = hw_data.get('Power_Consumption_Watts', {}).get('peak')

        # Se não houver dados, pula o modelo
        if ram_avg is None and power_avg is None:
            continue

        # Montagem da linha intercalando média e pico
        row = [
            model_name,
            format_ptbr(ram_avg),
            format_ptbr(ram_peak),
            format_ptbr(gpu_avg),
            format_ptbr(gpu_peak),
            format_ptbr(vram_avg),
            format_ptbr(vram_peak),
            format_ptbr(power_avg),
            format_ptbr(power_peak)
        ]
        
        csv_lines.append(";".join(row))

    output_path = 'resources_hardware.csv'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(csv_lines))
        
    print(f"CSV gerado com sucesso em: {output_path}")

if __name__ == "__main__":
    generate_hardware_csv()