FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

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

RUN pip install tqdm==4.67.1 wandb==0.15.4 scikit-learn==1.4.2 matplotlib==3.8.0 gdown==5.2.0

RUN echo "source /etc/bash_completion" >> /etc/bash.bashrc

WORKDIR /home/jmn

CMD ["bash"]
