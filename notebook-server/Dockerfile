FROM python:3
RUN apt-get update
RUN pip3 install --upgrade pip
RUN pip3 install notebook paho-mqtt
RUN mkdir /opt/notebooks
COPY ./Config.ipynb /opt/notebooks
COPY jupyter.sh .
ENTRYPOINT ["./jupyter.sh"]
