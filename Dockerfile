FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04
RUN apt update
RUN apt-get install -y python3 python3-pip

RUN pip install torch==1.4.0 torchvision==0.5.0

WORKDIR /DRIVE

COPY . /DRIVE

RUN pip install -r requirements.txt