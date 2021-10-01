# Build a plane Detector using Labeled Plane data
### Build a Docker container with TF, GPU Support, and Jupyter 


This directory contains a collection of [Jupyter Notebooks](notebooks) and [Scripts](scripts) that can be used to build datasets and train a model. The scripts are designed to automate most of the steps in the process. If you want to walk through the process yourself, checkout the notebooks.

## Requirements 
Since you will be training a model, an Nvidia GPU is required. In order for a Docker container to access the GPU, the Nvidia Container Toolkit needs to be installed. There are directions for doing that [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). You can test to make sure everything is installed correctly and working, with the following command:

````
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
````

## Build Docker Image
In the main directory of this repository, run the following command:
````
docker-compose build
````

## Launch the Docker Container
In the main directory of this repository, use Docker Compose to launch the container:

```bash
docker-compose up
```



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


# Running Scripts

To run the automated scripts, you will want to attach to a shell inside the container:

```bash
sudo docker exec -it ml-model_jupyter_1  /bin/bash
cd scripts
```

Now you are ready to start running scripts. Check out the documentation [here](scripts/README.md).

# Running Notebooks

Goto to the IP/domain name for the computer this is running on.... probably localhost. and port 8888 in a browser: `http://localhost:8888`

This will bring up the list of folders in the Jupyter client, go into the **notebooks** folder.

## Notebooks
Here is what the following notebooks help you do and the rough order you want to do them in:

*We use LabelBox to label our images. The free tier should support most tasks. Prior to starting, create a Labelbox Project and associated dataset. You will need the Labelbox project & dataset name, as well as a Labelbox API.


- **Create Voxel51 Dataset.ipynb** Run this first to load images from the *testing* directory into a Voxel51 dataset
- **Add FAA Data to Voxel51 Dataset.ipynb** Adds labels to the Voxel51 Dataset based on the FAA data
- **Upload from Voxel51 to Labelbox.ipynb** Sends the samples to Labelbox for annotation


### Labeling
After you have finished uploading the images, go to Labelbox and label the images. Then Export the labels as a JSON file. Download the JSON file and move it to the machine that is running the Docker container that is serving the Notebook.


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

