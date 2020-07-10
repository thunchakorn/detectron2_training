FROM python:3.8-slim-buster

WORKDIR /app

COPY ./ ./

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev

# Detectron2 prerequisites
RUN pip install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython 
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Development packages
RUN pip install opencv-python
RUN python -m pip install -r requirements.txt

# ENTRYPOINT ["python", "main.py"]