# Teachable Camera MQTT Broker

This is an MQTT Broker that runs locally on the Coral Board. It builds on this [Docker Image](https://github.com/mje-nz/rpi-docker-mosquitto), which is a version of the Eclipse Mosquitto broker which has been compiled for ARM.

A config file is added to enable support for sending MQTT messages over WebSockets. This allows the Javascript MQTT client in the WebApp to connect directly to this MQTT Broker and receive messages.
