FROM ultralytics/ultralytics:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    git \
    wget \
    vim \
    nano \
    gedit \
    curl \
    unzip \
    zip \
    xz-utils \
    mesa-utils \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglu1-mesa-dev \
    libxrender1 \
    libsm6 \
    libxext6 \
    bash-completion \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

RUN pip install tqdm wandb scikit-learn matplotlib gdown

RUN echo "source /etc/bash_completion" >> /etc/bash.bashrc

WORKDIR /home/jmn

CMD ["bash"]
