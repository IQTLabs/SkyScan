printf "\n\nCreating SkyScan Project Folder..."
mkdir ./SkyScan
cd SkyScan

printf "\n\nDownload Docker-Compose SkyScan Files..."
curl -fsSL https://raw.githubusercontent.com/IQTLabs/SkyScan/main/docker-compose.yml -o docker-compose.yml

if [ ! -d "mqtt/" ]; then
    mkdir mqtt;
fi
pushd mqtt
curl -fsSL https://raw.githubusercontent.com/IQTLabs/SkyScan/main/mqtt/docker-compose.mqtt.yml -o docker-compose.mqtt.yml
curl -fsSL https://raw.githubusercontent.com/IQTLabs/SkyScan/main/mqtt/ca.env -o ca.env
popd

curl -fsSL https://raw.githubusercontent.com/IQTLabs/SkyScan/main/.env-example -o .env
curl -fsSL https://raw.githubusercontent.com/IQTLabs/SkyScan/main/container.env-example -o container.env

printf "\n\nDownload Aircraft Database..."
mkdir ./data
curl -fsSL https://opensky-network.org/datasets/metadata/aircraftDatabase.csv -o ./data/aircraftDatabase.csv

printf "\n\nDownload Docker Containers..."
docker-compose -f docker-compose.yml -f mqtt/docker-compose.mqtt.yml pull

printf "\n\n\n\n--------------------------------------------------------------------"
printf "\n1: Configure network interface to same as Axis Camera (192.168.1.x)"
printf "\n2: Configure .env file appropriately"
printf "\n3: docker-compose -f docker-compose.yml -f mqtt/docker-compose.mqtt.yml up"
printf "\n--------------------------------------------------------------------\n\n"
