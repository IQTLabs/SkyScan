FROM python:3
RUN apt-get update
RUN pip3 install --upgrade pip
RUN pip3 install jupyter paho-mqtt
RUN mkdir /opt/notebooks
RUN echo "testing!"
COPY ./Config.ipynb /opt/notebooks
ENTRYPOINT jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root --NotebookApp.password='argon2:$argon2id$v=19$m=10240,t=10,p=8$CY9XaY3kdvPjn8nIwsBFvA$ecQYiVCZNsQu3KzM4YyhfU0etIXG8hFmIzeI4ROunGQ'
