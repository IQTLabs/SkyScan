# A Few Scripts to Rule Them All

This folder contains scripts that enable an operator or analyst
to run data- and model-related SkyScan functions from a script
rather than a series of Jupyter notebooks.

## Using the Scripts

First, run the Docker container.

```
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


## To run tests:

```
pytest
```
