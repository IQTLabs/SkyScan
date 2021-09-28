# Build a plane Detector using Labeled Plane data
### Build a Docker container with TF, GPU Support, and Jupyter 

## Requirements 
Since you will be training a model, an Nvidia GPU is required. In order for a Docker container to access the GPU, the Nvidia Container Toolkit needs to be installed. There are directions for doing that [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). You can test to make sure everything is installed correctly and working, with the following command:

````
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
````

## Build Docker Image
In the main directory of this repository, run the following command:
````
sudo docker build -t plane-jupyter .
````

## Launch the Docker Container
In the main directory of this repository, the following command will launch the container.

````
sudo docker run --name  plane-jupyter \
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
Check the logs as it is starting up and look for a token. Select and copy it because you will need to later.


Then go to to the IP/domain name for the computer this is running on.... probably localhost. and port 8888 in a browser: `http://localhost:8888`

Once there, paste in the token...

This will bring up the list of folders, go into notebooks.

## Directories
The following directories will be created when the container is launched for the first time. 
They are mapped into the container under **/tf/**.
Here is what they are used for:
- **model-export** Trained models are exported here
- **dataset-export** When a dataset is exported
- **notebooks** This is where the wonderful notebooks are stored, this way if you make changes to them they serve the container restarting
- **testing** This is for passing in raw images that were captured
- **fiftyone-db** This gets mapped to the location where the Voxel51 DB for the datasets is saved. This lets them stay around between container restarts.
- **models** The [TF Model repo](https://github.com/tensorflow/models) gets installed here. 




## Notebooks
Here is what the following notebooks help you do and the rough order you want to do them in:

*We use LabelBox to label our images. The free tier should support most tasks. Prior to starting, create a Labelbox Project and associated dataset. You will need the Labelbox project & dataset name, as well as a Labelbox API.


- **Create Voxel51 Dataset.ipynb** Run this first to load images from the *testing* directory into a Voxel51 dataset
- **Add FAA Data to Voxel51 Dataset.ipynb** Adds labels to the Voxel51 Dataset based on the FAA data
- **Upload from Voxel51 to Labelbox.ipynb** Sends the samples to Labelbox for annotation


### Labeling
After you have finished uploading the images, go to Labelbox and label the images. Then export the labels as a JSON file. Download the JSON file and move it to the machine that is running the Docker container that is serving the Notebook.


### For classification
- **Export from Labelbox to Voxel51.ipynb** Export the annotations from Labelbox
- **Train Plane Classification Model.ipynb**
- **Add Plane Classification to Voxel51 Dataset.ipynb**

### For Object Detection
- **Export from Labelbox to Voxel51.ipynb** Export the annotations from Labelbox
- **Train Plane Detection Model.ipynb**
- **Add Object Detection to Voxel51.ipynb**

- **Examine TFRecord.ipynb** You can use this if you want to check out what is in a TFRecord

### Attach to a running container
If you wish to access the running version of the container and poke around on the command line inside it, use the following command:
````
sudo docker exec -it ml-model_jupyter_1  /bin/bash
````


### Tensorboard
If you wish to monitor the progress of a model being trained, Tensorboard provides a nice visualization of different metrics. To launch it, use the command above to attach to the running container and then run the following:
````
tensorboard --logdir=/tf/training/ --bind_all
````
If you goto **port 6006** of the machine where the container is running in a browser, the Tensorboard app should pop up.

````
python object_detection/model_main_tf2.py \
    --pipeline_config_path=/tf/models/research/deploy/pipeline_file.config  \
    --model_dir=/tf/training/d0_plane_detect \
    --checkpoint_dir=/tf/training/d0_plane_detect \
    --alsologtostderr
````

## Edge-TPU Models
The TF 2 Object Detection API may not be able to generate models that can be compiled to run on the Edge TPU / Coral. 
https://github.com/tensorflow/models/issues/8935

The fall back plan is to use the TF1 method to generate a model that can work with the EdgeTPU, described here:

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
