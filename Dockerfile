FROM nvcr.io/nvidia/cuda:10.1-devel-ubuntu18.04

WORKDIR /DRIVE

COPY . /DRIVE

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda create -n pyRL python=3.7 -y && \
    echo "source activate pyRL" > ~/.bashrc
ENV PATH /opt/conda/envs/pyRL/bin:$PATH

# RUN conda run -n pyRL conda install cudatoolkit=10.1 -c pytorch
# RUN conda run -n pyRL pip install -r requirements.txt