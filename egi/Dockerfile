FROM debian

RUN apt update && \
    apt install -y python3 python3-pip && \
    pip3 install paho-mqtt

RUN mkdir -p /app/
WORKDIR /app
ADD *.py /app/

#ENTRYPOINT python3 /tmp/egi.py

#docker run -d --restart unless-stopped --network=host -v /home/pi/:/tmp/ --name lamp docker-registry.iqt.org/mission-capabilities/rpi-stats-reporting/lamp-control
