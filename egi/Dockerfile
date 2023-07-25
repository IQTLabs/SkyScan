FROM debian

RUN apt update && \
    apt install -y python3 python3-pip gpsd-clients gpsd

RUN mkdir -p /app/
WORKDIR /app
ADD *.txt /app/ 
RUN pip3 install --break-system-packages -r requirements.txt
ADD *.py /app/

ENTRYPOINT sh -c "gpsd /dev/ttyACM0 -F /var/run/gpsd.sock && bash"

#docker run -it --device=/dev/ttyACM0 skyscan_egi
#python3 egi_mqtt.py -h