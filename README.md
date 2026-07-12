# 📷 Distortions Project

Este repositório contém uma suíte de modelos de Deep Learning e scripts voltados para o treinamento e teste de redes neurais (MobileNet, ResNet, YOLO e modelos de Fusão - Early/Late) aplicadas à **classificação de ruídos e distorções em imagens**.

O projeto visa categorizar as imagens de acordo com o tipo de degradação presente, operando sobre as seguintes classes:
* **AWGN** (Additive White Gaussian Noise)
* **Blur** (Desfoque)
* **Contrast** (Alterações de Contraste)
* **Fnoise** (Ruído de Frequência)
* **JPEG** (Artefatos de compressão JPEG)
* **JPEG2000** (Artefatos de compressão JPEG2000)
* **SRC** (Imagens originais/limpas - *pristine*)

O projeto foi construído de forma modular, permitindo o fácil intercâmbio entre diferentes arquiteturas de modelos, datasets e estratégias de fusão de dados.

---

## 💻 Pré-requisitos

Para rodar este projeto localmente, você precisará atender aos seguintes requisitos de sistema e software:

* **Python:** Versão `3.12`
* **Hardware (Treinamento):** GPU com no mínimo **16 GB de VRAM** (exigência para suportar o tamanho dos lotes e as arquiteturas das redes).
* **Docker:** (Opcional) Caso prefira rodar via contêiner para isolamento do ambiente.

---

## ⚙️ Instalação e Configuração

**1. Clone o repositório:**
```bash
git clone https://github.com/jefferson-norberto2/distortions.git -b dissertacao
cd distortions
```

**2. Configure as variáveis de ambiente:**
O projeto utiliza um arquivo `.env` para gerenciar variáveis sensíveis e configurações (como caminhos de pastas). Crie o seu baseando-se no arquivo de exemplo fornecido:
```bash
cp sample.env .env
```
*(Lembre-se de editar o arquivo `.env` gerado conforme as necessidades e caminhos do seu ambiente).*

**3. Instale as dependências:**
Recomenda-se o uso de um ambiente virtual (`venv` ou `conda`).
```bash
pip install -r requirements.txt
```

---

## 🚀 Como Usar

O projeto possui dois arquivos facilitadores na raiz do repositório para centralizar a execução dos treinamentos e testes: `main_train.py` e `main_test.py`.

### Realizando Treinamentos

Para iniciar o treinamento, execute o `main_train.py` passando o nome da arquitetura desejada como argumento. Se nenhum argumento for passado, o modelo padrão (`mobilenet`) será utilizado.

**Sintaxe:**
```bash
python main_train.py [modelo]
```

**Modelos disponíveis:**
* `mobilenet` (Padrão)
* `resnet`
* `early` (Estratégia de Early Fusion)
* `late` (Estratégia de Late Fusion)
* `yolo`

**Exemplo de uso:**
```bash
python main_train.py resnet
```

### Realizando Testes/Inferências

A lógica para os testes segue exatamente o mesmo padrão, utilizando o arquivo facilitador `main_test.py`:

```bash
python main_test.py yolo
```

---

## 🐳 Execução via Docker

Se você preferir isolar o ambiente e evitar problemas de dependências locais, o projeto já vem configurado com scripts facilitadores para Docker. 

> **Aviso:** Estes scripts devem ser executados sempre a partir da **raiz** do repositório.

**1. Construir a imagem (Build):**
```bash
bash .docker/build.sh
```

**2. Rodar o contêiner (Run):**
```bash
bash .docker/run.sh
```

---

## 🗂️ Estrutura do Projeto

Abaixo está o descritivo da estrutura principal de pastas e arquivos do repositório:

* **`distortions/`**: Pacote principal contendo todo o código-fonte do sistema.
    * **`dataset/`**: Classes customizadas para o carregamento dos dados (`single_dataset.py`, `dual_dataset.py`, `yolo_dataset.py`).
    * **`model/`**: Definição das arquiteturas das redes neurais utilizadas (MobileNet, ResNet e módulos de fusão).
    * **`scripts/`**: Scripts modulares separados entre `train/` e `test/` de acordo com a arquitetura.
    * **`utils/`**: Ferramentas auxiliares, abrangendo desde a geração de distorções até medição de GFLOPs/hardware, manipulação de arquivos e sistema de logs (`dual_logger.py`).
    * **`extras/`**: Scripts utilitários extras. Destaca-se o **`test_wild.py`**, um script em desenvolvimento voltado para avaliação do modelo em um dataset externo com imagens capturadas do mundo real (*in the wild*), testando a generalização do projeto além do escopo atual.
* **`main_train.py` / `main_test.py`**: Pontos de entrada unificados que facilitam o acionamento dos módulos da pasta `scripts/`.
* **`sample.env`**: Exemplo do mapeamento de variáveis de ambiente necessárias.
* **`requirements.txt`**: Lista de dependências e bibliotecas do Python.