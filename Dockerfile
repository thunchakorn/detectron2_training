FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev nano && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install tensorboard
RUN pip install torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html

RUN pip install 'git+https://github.com/facebookresearch/fvcore'

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Detectron2
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /app/

COPY train_utils.py ./
COPY main.py .
COPY ./ ./
# install requirements libraries for training
RUN python -m pip install -r requirements.txt
