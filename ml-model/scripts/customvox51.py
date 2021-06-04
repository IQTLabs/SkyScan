"""custom functionality related to voxel51 databases"""

import logging
import os

import pandas as pd
import fiftyone as fo

# pylint: disable=C0330


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


def add_sample_images_to_voxel51_dataset(image_list, dataset):
    """Add sample images to a voxel51 dataset.

    # TODO: Add check to make sure you can't add the same image twice

    Args:
        image_list - list of image data dicts
        dataset - a voxel51 dataset object

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

        dataset.add_sample(sample)

    # return modified dataset
    return dataset


def add_faa_data_to_voxel51_dataset(voxel51_dataset_name, faa_dataset_path):
    """Add FAA data to each entry in voxel51 dataset.

    Args:
        voxel51_dataset (str) - the voxel51 dataset name
        faa_dataset_path - path to FAA dataset csv

    Returns:
        dataset (voxel51 dataset object)
    """
    planes = pd.read_csv(faa_dataset_path, index_col="icao24")
    dataset = fo.load_dataset(voxel51_dataset_name)
    for row in dataset:
        # render plane_id in lowercase letters
        plane_id = row["icao24"].label.lower()
        try:
            plane = planes.loc[plane_id.lower()]
            # Check for valid row with all columns present
            if plane.size == 26:
                if isinstance(plane["model"], str):
                    row["model"] = fo.Classification(label=plane["model"])
                if isinstance(plane["manufacturername"], str):
                    row["manufacturer"] = fo.Classification(
                        label=plane["manufacturername"]
                    )
                # TODO: (for Luke) isn't there some other label Adam requested?
                row.save()
            else:
                logging.info("Invalid row with row size of %s", plane.size)
        except KeyError:
            logging.info("FAA Data entry not found for: %s", plane_id)

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
        norm_model (str) - a standardized model name
    """
    # TODO: probably convert this into a json dict structure that can then be more easily
    # understood within the code
    norm_model = None

    if model in ["A319-112", "A319-115", "A319-132"]:
        norm_model = "A319"

    if model in [
        "A320 232",
        "A320 232SL",
        "A320-211",
        "A320-212",
        "A320-214",
        "A320-232",
        "A320-251N",
        "A320-271N",
    ]:
        norm_model = "A320"

    if model in ("A321-211", "A321-231", "A321-271NX", "Airbus A321-231(SL)"):
        norm_model = "A321"

    if model in ("A330 243", "A330-243F"):
        norm_model = "A330"

    if model in ("737-71B", "737-724", "737-73V", "737-752", "737-7H4", "737-7Q8"):
        norm_model = "737-700"

    if model in (
        "737-800",
        "Boeing 737-852",
        "737-823",
        "737-824",
        "737-832",
        "737-83N",
        "737-84P",
        "737-890",
        "737-8EH",
        "737-8H4",
        "737NG 823/W",
        "737NG 852/W",
        "737NG 85P/W",
        "737NG 86N/W",
        "737NG 8V3/W",
    ):
        norm_model = "737-800"

    if model in ("737-900ER", "737-924ER", "737-932ER"):
        norm_model = "737-900"

    if model == "747-48EF":
        norm_model = "747-400"

    if model in ("757-231", "757-232", "757-251"):
        norm_model = "757-200"

    if model == "767 330ER/W":
        norm_model = "767-300"

    if model == "777-223":
        norm_model = "777-200"

    if model == "787-8":
        norm_model = "787-800"

    if model in ("787-9 (Boeing)", "BOEING 787-9 Dreamliner"):
        norm_model = "787-800"

    if model in ("45", "60"):
        norm_model = "Learjet 45/60"

    if model in (
        "510",
        "Citation Excel",
        "Citation Sovereign+",
        "525",
        "550",
        "560",
        "680",
        "750",
        "525A",
        "525B",
        "525C",
        "560XL",
        "680A",
    ):
        norm_model = "Cessna Jet"

    if model in ("CL-600-2B16", "BD-100-1A10", "BD-700-1A11"):
        norm_model = "Bombardier Challanger"

    if model == "CL-600-2C10":
        norm_model = "CRJ700"

    if model == "CL-600-2C11":
        norm_model = "CRJ550"

    if model in ("CL-600-2D24", "CRJ 900 LR NG", "CRJ-900"):
        norm_model = "CRJ900"

    if model in (
        "ERJ 170-100 SE",
        "ERJ 170-100SU",
        "ERJ 170-200 LR",
        "ERJ 190-100 IGW",
        "EMB-190 AR",
    ):
        norm_model = "ERJ-170"

    if model in ("EMB-135BJ", "EMB-145LR"):
        norm_model = "EMB-135"

    if model in ("EMB-505", "EMB-545"):
        norm_model = "EMB-505"

    if model == "PA-23-250":
        norm_model = "Piper PA-23"

    if model == "PC-12/47E":
        norm_model = "Pilatus PC-12"

    if model in (
        "FALCON 10",
        "FALCON 2000",
        "FALCON 50",
        "FALCON 7X",
        "FALCON 900 EX",
        "FALCON 900EX",
    ):
        norm_model = "Falcon"

    if model in (
        "G-IV",
        "G-V",
        "GALAXY",
        "GIV-X (G450)",
        "GULFSTREAM 200",
        "GV-SP (G550)",
        "GVI(G650ER)",
    ):
        norm_model = "Gulfstream"

    if model in ("HAWKER 800XP", "HAWKER 900XP"):
        norm_model = "Hawker"

    if model == "SF50":
        norm_model = "Cirrus"

    if model == "PRESSURIZED LANCR IV":
        norm_model = "Lancair IV"

    if model == "B300":
        norm_model = "King Air"

    return norm_model
