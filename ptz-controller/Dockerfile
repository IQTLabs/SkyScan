FROM iqtlabs/edgetech-core:latest
RUN apt-get update && apt-get install -y \
  build-essential \
  && rm -rf /var/lib/apt/lists/*
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
COPY ./base_mqtt_pub_sub.py .
COPY ./ptz_controller.py .
COPY ./ptz_utilities.py .
RUN mkdir -p /data/tosort
ENTRYPOINT [ "python3", "ptz_controller.py" ]
