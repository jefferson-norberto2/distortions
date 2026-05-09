import yaml
from pathlib import Path
from distortions.utils.functions import extract_model_parts

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
        'mobilenet_V1', 'mobilenet_V2', 'mobilenet_V3_small', 'mobilenet_V3_large',
    ]
    colors = ['RGB', 'LAB', 'HSV']

    header = "model;test_accuracy;cross_accuracy"
    
    for color in colors:
        csv_lines = [header]
        
        for model_name in models:
            family, version = extract_model_parts(model_name)
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

            # --- 3. Montagem da Linha ---
            row = [
                model_name, # Aqui deixamos só o nome do modelo, pois a cor já estará no nome do arquivo
                format_ptbr(test_acc),
                format_ptbr(cross_acc)
            ]
            
            csv_lines.append(";".join(row))

        # --- 4. Salvando o CSV da Cor Atual ---
        # Só salva o arquivo se tiver mais que o cabeçalho
        if len(csv_lines) > 1:
            output_path = f'accuracy_{color}.csv'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(csv_lines))
            print(f"CSV gerado com sucesso: {output_path}")

if __name__ == "__main__":
    generate_csv_by_color()