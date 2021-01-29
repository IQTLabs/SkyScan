# Build a plane Detector using Labeled Plane data
### Build a Docker container with TF, GPU Support, and Jupyter 
In the main directory of this repository, run the following command:
````
sudo docker build -t plane-jupyter .
````


````
sudo docker run -v $PWD/dataset:/tf/dataset -v $PWD/notebooks:/tf/notebooks -v $PWD/testing:/tf/testing -v $PWD/export:/tf/export -v $PWD/fiftyone-db:/root/.fiftyone --gpus all -p 8888:8888  -p 6006:6006 -p 5151:5151 -it --name  plane-jupyter --rm plane-jupyter 
````