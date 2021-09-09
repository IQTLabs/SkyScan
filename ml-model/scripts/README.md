# A Few Scripts to Rule Them All

This folder contains scripts that enable an operator or analyst to run data- and model-related SkyScan
functions from a script rather than a series of Jupyter notebooks. 

What do these scripts do? They largely implement machine learning-related functionality so that a
SkyScan user can build trained models that do plane detection and classification. Like many ML
application, much of the code is related to data preparation rather than simply model training or
inference.

## Using the Scripts

First, run the Docker container.

```
sudo docker exec -it ml-model_jupyter_1 /bin/bash
```

Next, install required dependencies.

```
pip install -r requirements.txt
```

To get help, type:

```
python main.py --help
```

There are a number of different functionalities that can be called from the script. Each
functionality requires any necessary input values to be provided via the config.ini
configuration file.

### Prepare data for training and analysis

To prepare the data for analysis, first provide a 'dataset_name' value and an 'image_directory'
file location value in the configuration file. Then run:

```
python main.py --prep
```

This command creates a voxel51 dataset based on the plane images you provide and enriches the
plane data with information from publicly-accessible FAA data.

### To normalize the plane model data

Because the character string identifying the plane model can vary widely even for the same plane model, this command
attempts to create a standardized model identifier for each plane. 

```
python main.py --normalize
```

### Upload training or evaluation dataset to Labelbox

To upload training or evaluation images to Labelbox for manual labeling, use the appropriate command below.

```
python main.py --upload_train
```

```
python main.py --upload_eval
```

Users must provide in the config their Labelbox API key, the Labelbox dataset name, the Labelbox project name, and the
name of the local dataset to be uploaded. The user will first need to create a Labelbox account, project, and dataset.

### Resume uploading training or evaluation dataset to Labelbox

Similar to the command above but in the event that the upload is disrupted or paused. Use one of these commands:

```
python main.py --resume_upload_train
```

```
python main.py --resume_upload_eval
```

The same configuration arguments as above are used.

### Download annotated dataset from Labelbox

After using Labelbox to do hand annotation, you then then merge the annotations with the Voxel51 dataset. First, download
the labels from Labelbox in a JSON format. Then run:

```
python main.py --download
```

The configuration file must contain values for the local Voxel51 dataset name and also the path of the JSON exported from Labelbox.

### Train a detection model

Train a deep learning model to do detection of plane objects.

```
python main.py --train
```

The configuration file must contain the dataset_name, the model's training_name, the base_model, and the num_train_steps.

### Export the model

Export the trained deep learning model.

```
python main.py --export_model
```

The configuration file must contain the dataset_name, the model's training_name, and the model's base_model.

### Make predictions with trained model

To use a trained deep learning model to make predictions, use:

```
python main.py --predict
```

The configuration file must contain the dataset_name, the model's training_name, and the prediction_field.

## A Potential Sequence of Commands

To help the user gain a sense of potential command sequences that could be useful, we provide one example
below.

First, enter the Docker container.

```
sudo docker exec -it ml-model_jupyter_1 /bin/bash
```

Next, install required dependencies.

```
pip install -r requirements.txt
```

After entering required values in the configuration file (e.g. 'dataset_name' = 'test' and 'image_directory' = 'foo'),
run the command:

```
python main.py --prep
```

The perform normalization on the plane model data.

```
python main.py --normalize
```

After creating a Labelbox account, API key, project name, and dataset name and then entering required values in the configuration file (e.g. api_key = 'password123', "labelbox" dataset_name = 'labelbox_data', project_name = 'labelbox_project', "filenames" dataset_name = 'test'), run a command to upload
the normalized data to Labelbox.

```
python main.py --upload_train
```

After performing labeling in Labelbox, export the results as a JSON. After entering required values in the configuration file (e.g "file_names" "dataset_name"= 'test', exported_json_path = 'foo/bar.json'), then merge the labels into the existing voxel51 dataset. Run the following command:

```
python main.py --download
```

After entering required values in the configuration file (e.g. 'dataset_name' = 'test', training_name = 'test_model', base_model = 'efficientdet-d0', and num_train_steps = 40000), then run this command to train a model:

```
python main.py --train
```        
                       
## To run tests:

```
pytest
```
