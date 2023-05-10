#! /bin/sh
unset HISTFILE

touch /etc/mosquitto/passwd
touch /var/log/mosquitto.log
chown mosquitto:mosquitto /var/log/mosquitto.log
chown mosquitto:mosquitto /etc/mosquitto/passwd
#mosquitto_passwd -b /etc/mosquitto/passwd $(cat /run/secrets/mqtt_user.txt) $(cat /run/secrets/mqtt_passwd.txt)
mosquitto_passwd -b /etc/mosquitto/passwd $(cat /run/secrets/mqtt_user) $(cat /run/secrets/mqtt_passwd)
/usr/sbin/mosquitto -v -c /etc/mosquitto/mosquitto.conf

# test with mosquitto_pub -h mqtt-server -u $(cat /run/secrets/mqtt_user) -P $(cat /run/secrets/mqtt_pass) -t "test" -m "Hats!"

#systemctl enable mosquitto.service
#apk add certbot mosquitto mosquitto-clients 
#apk add vim

#certbot certonly --standalone --preferred-challenges http -d mqtt-skyscan
#mosquitto_passwd "$(cat /run/secrets/login_info_MQTT)"
#cp mqtt.conf /etc/mosquitto/conf.d/default.conf 
#vi /etc/mosquitto/conf.d/default.conf 

#ufw allow 8883
#ufw allow 443

#vi /etc/letsencrypt/renewal/mqtt-skyscan.conf
#certbot renew --dry-run