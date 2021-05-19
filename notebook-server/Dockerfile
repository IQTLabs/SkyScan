FROM python:3

RUN apt-get update
#RUN apt-get install python3-matplotlib
#RUN apt-get install python3-scipy
RUN pip3 install --upgrade pip
RUN pip3 install jupyter
RUN pip3 install paho-mqtt
RUN mkdir /opt/notebooks
COPY ./Config.ipynb /opt/notebooks

ENTRYPOINT jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
