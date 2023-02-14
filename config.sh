printf "\n\nDownload Docker-Compose SkyScan Files..."
curl -fsSL https://raw.githubusercontent.com/IQTLabs/SkyScan/main/docker-compose.yml -o docker-compose.yml
curl -fsSL https://raw.githubusercontent.com/IQTLabs/SkyScan/main/env-example -o .env

printf "\n\nDownload Aircraft Database..."
mkdir ./data
curl -fsSL https://opensky-network.org/datasets/metadata/aircraftDatabase.csv -o ./data/aircraftDatabase.csv

printf "\n\nDownload Docker Containers..."
docker-compose pull

printf "\n\n\n\n--------------------------------------------------------------------"
printf "\n1: Configure network interface to same as Axis Camera (192.168.1.x)"
printf "\n2: Configure .env file appropriately"
printf "\n3: docker-compose up"
printf "\n--------------------------------------------------------------------\n\n"
