FROM python:3.9.15-bullseye
LABEL org.label-schema.schema-version 1.0
LABEL org.label-schema.vendor IQT Labs
LABEL org.label-schema.name SkyScan Edge AI

# Dependencies
RUN apt-get update && apt-get upgrade --assume-yes && apt-get install -y zip htop screen libgl1-mesa-glx
RUN python --version

# YOLOv7
RUN git clone https://github.com/WongKinYiu/yolov7.git /yolov7
WORKDIR /yolov7
RUN sed -i '/thop/d' requirements.txt
RUN sed -i '/torchvision/d' requirements.txt
RUN sed -i '/torch/d' requirements.txt

RUN pip install torch==1.8.1 -f https://download.pytorch.org/whl/torch/ 
RUN pip install torchvision==0.9.1 -f https://download.pytorch.org/whl/torchvision/
RUN pip install -r requirements.txt

# SkyScan Edge AI
ADD sort.py ./
ADD datasets2.py utils/
RUN mkdir /data

# Run
CMD ["bash", "-c", "python sort.py \
--weights data/weights/localizer.pt \
--agnostic-nms --nosave --conf 0.25 --img-size 640 --device cpu \
--source-dir ../data/tosort \
--plane-dir ../data/plane \
--noplane-dir ../data/noplane \
--log-dir ../data/log \
--save-json"]

