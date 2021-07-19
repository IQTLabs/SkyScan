#!/usr/bin/env bash

# commands to install base training configuration file

config_file_url=$1
cd /tf/models/research/deploy
wget $config_file_url
