# Build a plane Detector using Labeled Plane data
### Build a Docker container with TF, GPU Support, and Jupyter 
In the main directory of this repository, run the following command:
````
sudo docker build -t plane-jupyter .
````


````
sudo docker run --name  plane-jupyter \
-v $PWD/dataset:/tf/dataset \
-v $PWD/notebooks:/tf/notebooks \
-v $PWD/testing:/tf/testing \
-v $PWD/export:/tf/export \
-v $PWD/fiftyone-db:/root/.fiftyone \
-v $PWD/models:/tf/models \
-v $PWD/training:/tf/training --gpus all \
-p 8888:8888  -p 6006:6006 -p 6007:6007 -p 5151:5151 \
-it --rm plane-jupyter 
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


## TF 1



https://coral.ai/docs/edgetpu/retrain-detection/#requirements

sudo docker run --name edgetpu-detect \
--rm -it --privileged -p 6006:6006 \
-v $PWD/tf1:/tensorflow/models/research/learn_pet \
-v $PWD/export:/tensorflow/models/research/export \
detect-tutorial-tf1


sudo docker build . -t detect-tutorial-tf1 -f Dockerfile.tf1

# Run this from within the Docker container (at tensorflow/models/research/):
./prepare_checkpoint_and_dataset.sh --network_type mobilenet_v2_ssd --train_whole_model false

NUM_TRAINING_STEPS=500 && \
NUM_EVAL_STEPS=100

# From the /tensorflow/models/research/ directory
./retrain_detection_model.sh \
--num_training_steps ${NUM_TRAINING_STEPS} \
--num_eval_steps ${NUM_EVAL_STEPS}