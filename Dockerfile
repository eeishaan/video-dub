FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

RUN apt update && apt install -y \
    git \
    wget \
    unzip \
    vim \
    ffmpeg \
    espeak-ng \
    && apt clean

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN wget https://huggingface.co/westbrook/SSR-Speech-English/resolve/main/English.pth?download=true -O /tmp/English.pth
RUN wget https://huggingface.co/westbrook/SSR-Speech-English/resolve/main/wmencodec.th?download=true -O /tmp/wmencodec.th
RUN wget https://huggingface.co/westbrook/SSR-Speech-English/resolve/main/vocab_en.txt -O /tmp/vocab_en.txt

WORKDIR /workspace
RUN git clone https://github.com/WangHelin1997/SSR-Speech.git
RUN python -c "import nltk; nltk.download('punkt')"

RUN conda update -n base -c defaults conda -y && \
    conda clean --all -y && \
    conda install -y -c conda-forge --override-channels montreal-forced-aligner==3.1.2

RUN mfa models download dictionary english_mfa && \
    mfa models download acoustic english_mfa && \
    mfa models download g2p english_us_mfa
