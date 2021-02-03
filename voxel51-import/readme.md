# Build a plane Detector using Labeled Plane data
### Build a Docker container with TF, GPU Support, and Jupyter 
In the main directory of this repository, run the following command:
````
sudo docker build -t plane-jupyter .
````


````
sudo docker run -v $PWD/dataset:/tf/dataset -v $PWD/notebooks:/tf/notebooks -v $PWD/testing:/tf/testing -v $PWD/export:/tf/export -v $PWD/fiftyone-db:/root/.fiftyone -v $PWD/models:/tf/models -v $PWD/training:/tf/training --gpus all -p 8888:8888  -p 6006:6006 -p 6007:6007 -p 5151:5151 -it --name  plane-jupyter --rm plane-jupyter 
````

## Setup
Images you want to run the model against go in the testing directory


# Notebooks


### Attach to a running container
If you wish to access the running version of the container and poke around on the command line inside it, use the following command:
````
sudo docker exec -it plane-jupyter  /bin/bash
````

### Tensorboard
If you wish to monitor the progress of a model being trained, Tensorboard provides a nice visualization of different metrics. To launch it, use the command above to attach to the running container and then run the following:
````
tensorboard --logdir=/tf/training/ --bind_all
````
If you goto **port 6006** of the machine where the container is running in a browser, the Tensorboard app should pop up.