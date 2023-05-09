#add-apt-repository ppa:certbot/certbot
#apt install certbot mosquitto mosquitto-clients

apk add certbot
apk add certbot mosquitto mosquitto-clients 
apk add nano
#these Ubuntu but should be in Alpine 1 and 2 how to install certbot in alpine linux, get dockerfile to build with certbot 

#ufw allow 80 
#expose these ports in the docker file 
certbot certonly --standalone --preferred-challenges http -d mqtt-skyscan
#mosquitto_passwd -c /etc/mosquitto/passwd user
# /b lets us pass the username and password in the command line, replacing -c
mosquitto_passwd "$(cat /run/secrets/login_info_MQTT)"
nano /etc/mosquitto/conf.d/default.conf 

ufw allow 8883
ufw allow 443

nano /etc/letsencrypt/renewal/mqtt-skyscan.conf
certbot renew --dry-run