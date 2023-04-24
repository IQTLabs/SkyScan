# Set environment variables
export COMPOSE_FILE_URL=
export ENV_FILE_URL=

echo "Installing skyscan on ${HOSTNAME} at $(pwd)"

echo "Configuring Mobian base"
bash <(curl -fsSL https://short.iqt.org/pinephonepro)

mkdir data
cd data
curl -O https://opensky-network.org/datasets/metadata/aircraftDatabase.csv
cd ..

# Make necessary folders
mkdir /flash
mkdir /flash/raw
mkdir /flash/coral
mkdir /flash/coral/plane
mkdir /flash/coral/noplane
mkdir /flash/coral/log
mkdir /flash/weights
mkdir /flash/edge
mkdir /flash/edge/plane
mkdir /flash/edge/noplane
mkdir /flash/edge/log
mkdir /flash/processed
mkdir /flash/processed/log

curl -O $COMPOSE_FILE_URL
curl -O $ENV_FILE_URL

docker-compose pull

echo "Installation complete. Run `docker-compose up` to start the system"
echo "Before running, ensure you have replaced the demo weights with your own trained weights"