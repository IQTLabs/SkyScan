#!/usr/bin/env bash

# commands to install FAA-related data

# store data assets in data directory
cd /tf/data

# remove any files previously downloaded
rm *.txt *.pdf *.zip

# donwload latest FAA-related datasets
wget https://registry.faa.gov/database/ReleasableAircraft.zip

# unzip
unzip ReleasableAircraft.zip

# rename key files for clarity
mv MASTER.txt faa_master.txt
mv ACFTREF.txt faa_aircraft_reference.txt
