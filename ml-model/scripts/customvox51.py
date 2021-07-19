"""custom functionality related to voxel51 databases"""

import json
import logging
import os
import subprocess

import pandas as pd
import fiftyone as fo

# pylint: disable=C0330,R0914


def build_image_list(file_path):
    """Create a list of plane data dicts.

    Extract plane data from from jpg filenames.

    The plane image filenames follow a strict naming convention.
    For instance, ac760d_194_50_11882_2021-05-13-14-13-42.jpg translates to
    ac760d - plane_id, aka ICAO 24
    194 - plane bearing
    50 - plane elevation
    11882 - plane distance
    ac760d_194_50_11882_2021-05-13-14-13-42 - external_id

    Args:
        file_path - Path to images

    Returns:
        (list) image_list - a list of plane dict objects
    """
    image_list = []
    logging.info("Building image list.")
    for folder, _, files in os.walk(file_path):
        for file in files:
            if file.endswith(".jpg"):
                # extract metadata for each plane from filename
                image_filename = os.path.basename(file)
                external_id = os.path.splitext(image_filename)[0]
                image_path = os.path.abspath(os.path.join(folder, file))
                plane_id = external_id.split("_")[0]
                plane_bearing = external_id.split("_")[1]
                plane_elevation = external_id.split("_")[2]
                plane_distance = external_id.split("_")[3]
                # place plane image data in a dict
                item = {
                    "file_path": image_path,
                    "external_id": external_id,
                    "bearing": plane_bearing,
                    "elevation": plane_elevation,
                    "distance": plane_distance,
                    "icao24": plane_id,
                }
                image_list.append(item)
    logging.info("Finished building image list.")

    return image_list


def create_voxel51_dataset(dataset_name):
    """Create a voxel51 dataset or load existing one.

    Args:
        dataset_name: name of the voxel51 dataset to create or load

    Returns:
        dataset (voxel51 dataset object)
    """
    # attempt to open dataset
    try:
        dataset = fo.Dataset(name=dataset_name)
        dataset.persistent = True
        logging.info("Created %s dataset", dataset_name)
    # If the dataset already exists, load it instead
    except ValueError:
        dataset = fo.load_dataset(name=dataset_name)
        logging.info("Dataset already exists. Loaded %s dataset", dataset_name)

    return dataset


def add_sample_images_to_voxel51_dataset(image_list, dataset, datasource_name=None):
    """Add sample images to a voxel51 dataset.

    # TODO: Add check to make sure you can't add the same image twice

    Args:
        image_list - list of image data dicts
        dataset - a voxel51 dataset object
        datasource_name - an optional string that allows for and identifying
                tag to be added to the batch of images being imported
    Returns:
        dataset (voxel51 dataset object)
    """
    for image in image_list:
        # create a voxel51 row/sample based on the path to the image
        sample = fo.Sample(filepath=image["file_path"])
        # add additional columns to the voxel51 dataset row
        sample["external_id"] = fo.Classification(label=image["external_id"])
        sample["bearing"] = fo.Classification(label=image["bearing"])
        sample["elevation"] = fo.Classification(label=image["elevation"])
        sample["distance"] = fo.Classification(label=image["distance"])
        sample["icao24"] = fo.Classification(label=image["icao24"])
        if datasource_name is not None and len(datasource_name) > 0:
            sample.tags.append(datasource_name)
        dataset.add_sample(sample)

    # return modified dataset
    return dataset


def add_faa_data_to_voxel51_dataset(
    voxel51_dataset_name, faa_master_dataset_path, faa_reference_dataset_path
):
    """Add FAA data to each entry in voxel51 dataset.

    Args:
        voxel51_dataset (str) - the voxel51 dataset name
        faa_master_dataset_path - path to FAA master dataset .txt
        faa_reference_dataset_path - path to FAA reference dataset .txt

    Returns:
        dataset (voxel51 dataset object)
    """
    subprocess.run("./install_faa_data.sh", check=True)

    # import master dataset and strip white space from beacon column
    planes_master = pd.read_csv(faa_master_dataset_path, index_col="MODE S CODE HEX")
    planes_master.index = planes_master.index.str.strip()
    planes_reference = pd.read_csv(faa_reference_dataset_path, index_col="CODE")
    dataset = fo.load_dataset(voxel51_dataset_name)

    for row in dataset:
        # render plane_id in lowercase letters
        plane_icao24 = row["icao24"].label.upper()
        # find plane model code associated with the icao24 code, i.e. mode s code hex
        try:
            model_code = planes_master.loc[
                planes_master.index == plane_icao24, "MFR MDL CODE"
            ].values[0]
        except IndexError:
            logging.info(
                "Plane ID not found in master dataset. Plane ID: %s", plane_icao24
            )
            continue

        # find reference row with all relevant model data
        plane_reference_row = planes_reference.loc[
            planes_reference["CODE"] == model_code
        ]
        # exract all relevant data from plane_reference_row
        # convert all fields to string
        manufacturer = str(plane_reference_row["MFR"].values[0])
        model_name = str(plane_reference_row["MODEL"].values[0])
        aircraft_type = str(plane_reference_row["TYPE-ACFT"].values[0])
        engine_type = str(plane_reference_row["TYPE-ENG"].values[0])
        num_engines = str(plane_reference_row["NO-ENG"].values[0])
        num_seats = str(plane_reference_row["NO-SEATS"].values[0])
        aircraft_weight = str(plane_reference_row["AC-WEIGHT"].values[0])

        # store values in voxel51 dataset row
        row["model_code"] = fo.Classification(label=model_code)
        row["manufacturer"] = fo.Classification(label=manufacturer)
        row["model_name"] = fo.Classification(label=model_name)
        row["aircraft_type"] = fo.Classification(label=aircraft_type)
        row["engine_type"] = fo.Classification(label=engine_type)
        row["num_engines"] = fo.Classification(label=num_engines)
        row["num_seats"] = fo.Classification(label=num_seats)
        row["aircraft_weight"] = fo.Classification(label=aircraft_weight)
        row.save()

    return dataset


def normalize_model_values(dataset):
    """Standardize plane model string values.

    The plane model string values received from ADS-B broadcasts
    are not standardized. An A319 model, for instance, could be
    represented as A319-112 or A319-115 or A39-132. This function
    helps standardize all model strings.

    Args:
        dataset - a voxel51 dataset object

    Returns:
        dataset - a voxel51 dataset object
    """
    # TODO: Need to add testing.
    # Loop thru each row of model column
    for sample in dataset.exists("model"):
        norm_model = normalize_single_model_value(sample["model"].label)
        if norm_model is not None:
            sample["norm_model"] = fo.Classification(label=norm_model)
            sample.save()
        else:
            logging.info("Match not found for: %s", sample["model"].label)

    return dataset


def normalize_single_model_value(model):
    """Standardize a single plane model string.

    Args:
        model (str) - a plane model name

    Returns:
        normalized_model_value (str) - a standardized model name
    """
    # json file storing plane model strings as key and standardized model
    # as value
    with open("plane_model_dict.json", "r") as file_path:
        plane_model_dict = json.load(file_path)

    # check for model value, if not present return None
    normalized_model_value = plane_model_dict.get(model, None)

    return normalized_model_value
