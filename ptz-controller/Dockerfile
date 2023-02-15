FROM iqtlabs/edgetech-core:latest
RUN apt install build-essential -y
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
ADD ./base_mqtt_pub_sub.py .
ADD ./ptz_controller.py .
ADD ./ptz_utilities.py .
ENTRYPOINT [ "python3", "ptz_controller.py" ]
