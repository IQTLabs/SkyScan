# Set environment variables
export COMPOSE_FILE_URL=https://raw.githubusercontent.com/meadej/SkyScan/main/docker-compose.yml
export ENV_FILE_URL=https://raw.githubusercontent.com/meadej/SkyScan/main/.env-example

echo "Installing skyscan on ${HOSTNAME} at $(pwd)"

echo "Configuring Mobian base"
bash <(curl -fsSL https://short.iqt.org/pinephonepro)


mkdir skyscan
cd skyscan
mkdir data
cd data
curl -O https://opensky-network.org/datasets/metadata/aircraftDatabase.csv
cd ..

# Make necessary folders
mkdir raw
mkdir coral
mkdir coral/plane
mkdir coral/noplane
mkdir coral/log
mkdir weights
mkdir edge
mkdir edge/plane
mkdir edge/noplane
mkdir edge/log
mkdir processed
mkdir processed/log

curl -O $COMPOSE_FILE_URL
curl -O $ENV_FILE_URL

docker-compose -f docker-compose.yml -f mqtt/docker-compose.mqtt.yml pull

echo "Installation complete. Run `docker-compose up` to start the system"
echo "Before running, ensure you have replaced the demo weights with your own trained weights"