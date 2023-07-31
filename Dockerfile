FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

RUN apt-get update && apt-get install -y \
    tree \
    vim \
    tmux \
    && apt-get clean