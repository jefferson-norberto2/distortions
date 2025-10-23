# ==========================
#  BASE IMAGE
# ==========================
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# ==========================
#  SYSTEM DEPENDENCIES
# ==========================
RUN apt update && apt upgrade -y && apt install -y \
    git \
    wget \
    vim \
    sudo \
    nano \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gedit \
    curl \
    unzip \
    xz-utils \
    zip \
    libglu1-mesa \
    libglu1-mesa-dev \
    mesa-utils \
    libxrender1 \
    libsm6 \
    libxext6 \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

# ==========================
#  CREATE NON-ROOT USER
# ==========================
ARG USERNAME=jmn
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME:123asd!" | chpasswd \
    # Allow sudo with password and 30-min cache
    && echo "$USERNAME ALL=(ALL) ALL" > /etc/sudoers.d/$USERNAME \
    && echo "Defaults:$USERNAME timestamp_timeout=30" >> /etc/sudoers.d/$USERNAME \
    && chmod 440 /etc/sudoers.d/$USERNAME

# ==========================
#  ENABLE BASH AUTOCOMPLETE
# ==========================
RUN echo '\n# Enable bash completion\nif [ -f /etc/bash_completion ]; then\n  . /etc/bash_completion\nfi' >> /home/$USERNAME/.bashrc

# ==========================
#  SET DEFAULT USER AND WORKDIR
# ==========================
USER $USERNAME
WORKDIR /home/$USERNAME

# ==========================
#  DEFAULT COMMAND
# ==========================
CMD ["bash"]
