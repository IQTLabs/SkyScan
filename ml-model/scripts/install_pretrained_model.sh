#!/usr/bin/env bash

# commands to install pretrained machine learning model
# wget must take the argument that specifies a URL from
# which to retrieve a model

url=$1
mkdir /tf/models/research/deploy/
cd /tf/models/research/deploy/
wget $url