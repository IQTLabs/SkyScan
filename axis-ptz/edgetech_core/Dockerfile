FROM python:3.11-slim

RUN pip install paho-mqtt==1.6.1
RUN pip install pytest==7.2.0
RUN apt update
RUN apt install screen -y

WORKDIR /root
ADD ./base_mqtt_pub_sub.py .
ADD ./base_mqtt_pub_sub_test.py .