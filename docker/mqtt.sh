#!/bin/sh

apk add certbot
apk add certbot mosquitto mosquitto-clients 
apk add nano

certbot certonly --standalone --preferred-challenges http -d mqtt-skyscan
mosquitto_passwd "$(cat /run/secrets/login_info_MQTT)"
nano /etc/mosquitto/conf.d/default.conf 

ufw allow 8883
ufw allow 443

nano /etc/letsencrypt/renewal/mqtt-skyscan.conf
certbot renew --dry-run