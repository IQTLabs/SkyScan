# edgetech-mqtt-compose
Docker Compose assets to utilize MQTTS in edgetech projects

## Instructions
1. create the following files:
  - .ca_password - contains the password to be used for the ca's root certificate
  - .mqtt_user - the username for mqtt clients
  - .mqtt_password - the password for mqtt clients to authenticate
1. If using from a different directory set the `MQTT_PREFIX` environment variable to be the path to the location of `docker-compose.mqtt.yml`
1. `docker compose -f <other_docker_compose_file> -f docker-compose.mqtt.yml up -d --build`