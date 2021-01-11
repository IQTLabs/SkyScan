FROM "mjenz/rpi-mosquitto"

COPY ./mosquitto.conf /etc/mosquitto/  

ENTRYPOINT /usr/sbin/mosquitto -c /etc/mosquitto/mosquitto.conf