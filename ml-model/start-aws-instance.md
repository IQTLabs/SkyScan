
## Start up instance
1. Turn on the B VPN
1. Goto SSO
1. Goto AWS SSO
1. Select AWS ACCount
1. Goto SkyScan Maganeent console
1. Goto EC2
1. Select the instace, select it, Start Instance
1. Click on the instance once it is running
1. Get the Private IP for the instance
1. Now goto the Terminal
1. `ssh -i "pth to .pem file" ubuntu@pirvate.ip.address`

## Start up Jupyter notebook
Go into the ml-model directory for skyscan
`cd ~/SkyScan-Private/ml-model`

Now launch
````
 docker run --name  plane-jupyter \
-v $PWD/model-export:/tf/model-export \
-v $PWD/dataset-export:/tf/dataset-export \
-v $PWD/notebooks:/tf/notebooks \
-v $PWD/testing:/tf/testing \
-v $PWD/media:/tf/media \
-v $PWD/fiftyone-db:/root/.fiftyone \
-v $PWD/models:/tf/models \
-v $PWD/training:/tf/training --gpus all \
-p 8888:8888  -p 6006:6006 -p 6007:6007 -p 5151:5151 \
-it --rm plane-jupyter 
````

1. copy the Token from the startup of the container
1. Open browser
1. Goto: `http://private.ip.adress:8888`
1. Open the **Notebooks** folder
