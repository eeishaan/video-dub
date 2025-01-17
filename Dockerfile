# FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    git \
    wget \
    unzip \
    vim \
    ffmpeg \
    espeak-ng \
    libgl1 \
    && apt clean

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN wget https://huggingface.co/westbrook/SSR-Speech-English/resolve/main/English.pth?download=true -O /tmp/English.pth
RUN wget https://huggingface.co/westbrook/SSR-Speech-English/resolve/main/wmencodec.th?download=true -O /tmp/wmencodec.th
RUN wget https://huggingface.co/westbrook/SSR-Speech-English/resolve/main/vocab_en.txt -O /tmp/vocab_en.txt
RUN wget https://huggingface.co/chunyu-li/LatentSync/resolve/main/whisper/tiny.pt?download=true -O /tmp/whisper_tiny.pt
RUN wget https://huggingface.co/chunyu-li/LatentSync/resolve/main/latentsync_unet.pt?download=true -O /tmp/latentsync_unet.pt
RUN mkdir -p /root/.cache/torch/hub/checkpoints/ mkdir checkpoints
RUN wget https://huggingface.co/chunyu-li/LatentSync/resolve/main/auxiliary/s3fd-619a316812.pth?download=true -O /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
RUN wget https://huggingface.co/chunyu-li/LatentSync/resolve/main/auxiliary/2DFAN4-cd938726ad.zip?download=true -O /root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip

WORKDIR /workspace
RUN git clone https://github.com/WangHelin1997/SSR-Speech.git ssr && \
    cd ssr && \
    git checkout 097c3c4f8c08acd7b431f095fa0f10e3151075c1 && \
    cd /workspace && \
    git clone https://github.com/bytedance/LatentSync.git LatentSync && \
    cd LatentSync && \
    git checkout 1fa59d9110d1f7b348e7006f590fecc9f9051834 && \
    touch scripts/__init__.py latentsync/__init__.py && \
    mkdir -p checkpoints/whisper && \
    ln -s /tmp/whisper_tiny.pt checkpoints/whisper/tiny.pt && \
    ln -s /tmp/latentsync_unet.pt checkpoints/latentsync_unet.pt && \
    cd /workspace

RUN python -c "import nltk; nltk.download('punkt')"

RUN conda update -n base -c defaults conda -y && \
    conda clean --all -y && \
    conda install -y -c conda-forge --override-channels montreal-forced-aligner==3.1.2

RUN mfa models download dictionary english_mfa && \
    mfa models download acoustic english_mfa && \
    mfa models download g2p english_us_mfa
