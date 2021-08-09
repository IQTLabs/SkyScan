"""custom functionality related to voxel51 databases"""

import json
import logging
import os
import subprocess
import math

import pandas as pd
import fiftyone as fo
from fiftyone import ViewField as F
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

def _tag_samples_by_icao24(dataset, icao24, tag, only_aircraft_detected=True):
    """Adds a tag to all samples with a matching ICAO24
    Args:
        dataset (Voxel51 Dataset): Dataset to work with 
        icao24 (string): the ICAO24 identifier to search for
        tag (string): Tag to add
    """    
    if only_aircraft_detected:
        view = dataset.filter_labels("icao24", (F("label") == icao24)).match(F("multi_class_detections.detections").length()>0)
    else:    
        view = dataset.filter_labels("icao24", (F("label") == icao24))
    for sample in view:
        sample.tags.append(tag)
        sample.save() 

def build_multi_class_train_eval_dataset(dataset_name):   
    dataset = fo.load_dataset(dataset_name)
    norm_models = dataset.distinct("norm_model.label")
    for norm_model in norm_models:
        view = dataset.filter_labels("norm_model", (F("label") == norm_model)).select_fields("icao24")
        unique_aircraft = view.distinct("icao24.label")
        num_unique_aircrarft = len(unique_aircraft)
        if num_unique_aircrarft > 1:
            _tag_samples_by_icao24(dataset,unique_aircraft[0], "multi_class_train")
            for icao24 in unique_aircraft[1:]:
                _tag_samples_by_icao24(dataset,icao24, "multi_class_eval")
            print("{}: {}".format(norm_model,len(unique_aircraft)))
            print("\tTrain:{}".format(unique_aircraft[0]))
            print("\tEval:{}".format(unique_aircraft[1:]))

def select_multi_class_train_eval_dataset(dataset_name, prediction_field, train_size):   

    dataset = fo.load_dataset(dataset_name)
    train_view = dataset.match_tags("multi_class_train")
    logging.iinfo("Removing existing multi_class_train tags")
    for sample in train_view:
        try:
            sample.tags = list(filter(lambda x: x != "multi_class_train", sample.tags))
            sample.save()
        except ValueError:
            pass
    logging.iinfo("Removing existing multi_class_eval tags")
    eval_view = dataset.match_tags("multi_class_eval")
    for sample in eval_view:
        try:
            sample.tags = list(filter(lambda x: x != "multi_class_eval", sample.tags))
            sample.save()
        except ValueError:
            pass

    norm_models = dataset.distinct("norm_model.label")
    for norm_model in norm_models:
        view = dataset.filter_labels("norm_model", (F("label") == norm_model)).match(F("auto_aug_predict_tiled.detections").length()>0).shuffle()
        print("{}: {}".format(norm_model,len(view)))
        if len(view) >= 200:
            for sample in view[:100]:
                sample.tags.append("multi_class_train")
                sample.save()
            for sample in view[100:]:
                sample.tags.append("multi_class_eval")
                sample.save()

def split_multi_class_train_eval_dataset(dataset_name):   
    """Splits the dataset into Training and Eval samples. For aircraft models with
    more than one example, the aircraft bodies will be divide, 75% to Train and 
    25% to Eval. The samples are separated using tags.

    Args:
        dataset_name ([type]): [description]
    """
    dataset = fo.load_dataset(dataset_name)
    train_view = dataset.match_tags("multi_class_train")
    logging.info("Removing existing multi_class_train tags")
    for sample in train_view:
        try:
            sample.tags = list(filter(lambda x: x != "multi_class_train", sample.tags))
            sample.save()
        except ValueError:
            pass
    logging.info("Removing existing multi_class_eval tags")
    eval_view = dataset.match_tags("multi_class_eval")
    for sample in eval_view:
        try:
            sample.tags = list(filter(lambda x: x != "multi_class_eval", sample.tags))
            sample.save()
        except ValueError:
            pass

    norm_models = dataset.distinct("norm_model.label")
    for norm_model in norm_models:
        view = dataset.filter_labels("norm_model", (F("label") == norm_model)).select_fields("icao24").shuffle()
        unique_aircraft = view.distinct("icao24.label")
        if len(unique_aircraft) > 1:
            train_aircraft = unique_aircraft[:math.floor(len(unique_aircraft)*.75)]
            eval_aircraft = unique_aircraft[math.floor(len(unique_aircraft)*.75):]
            print("{} Total: {} Train: {} Eval: {}".format(norm_model,len(unique_aircraft),len(train_aircraft),len(eval_aircraft)))     
            
            for icao24 in train_aircraft[:1]:
                _tag_samples_by_icao24(dataset,icao24, "multi_class_train", False)
            for icao24 in train_aircraft[1:]:
                _tag_samples_by_icao24(dataset,icao24, "multi_class_train", True)
        
            for icao24 in eval_aircraft:
                _tag_samples_by_icao24(dataset,icao24, "multi_class_eval", True)

def random_multi_class_train_eval_dataset(dataset_name):   
    """Splits the dataset into Training and Eval samples. For aircraft models with
    more than one example, the aircraft bodies will be divide, 75% to Train and 
    25% to Eval. The samples are separated using tags.

    Args:
        dataset_name ([type]): [description]
    """
    dataset = fo.load_dataset(dataset_name)
    train_view = dataset.match_tags("multi_class_train")
    logging.info("Removing existing multi_class_train tags")
    for sample in train_view:
        try:
            sample.tags = list(filter(lambda x: x != "multi_class_train", sample.tags))
            sample.save()
        except ValueError:
            pass
    logging.info("Removing existing multi_class_eval tags")
    eval_view = dataset.match_tags("multi_class_eval")
    for sample in eval_view:
        try:
            sample.tags = list(filter(lambda x: x != "multi_class_eval", sample.tags))
            sample.save()
        except ValueError:
            pass

    norm_models = dataset.distinct("norm_model.label")
    for norm_model in norm_models:
        view = dataset.filter_labels("norm_model", (F("label") == norm_model)).match(F("multi_class_detections.detections").length()>0).shuffle()
        unique_aircraft = view.distinct("icao24.label")

        train_count = math.floor(len(unique_aircraft)*.75)
        eval_count = math.floor(len(unique_aircraft)*.75)
        for sample in view[:train_count]:
            sample.tags.append("multi_class_train")
            sample.save() 
        for sample in view[train_count:]:
            sample.tags.append("multi_class_eval")
            sample.save() 

        print("{} Total: {} Train: {} Eval: {}".format(norm_model,len(unique_aircraft),len(train_aircraft),len(eval_aircraft)))   

    view = dataset.match(F("multi_class_detections.detections").length()==0).take(250)
    for sample in view:
        sample.tags.append("multi_class_train")
        sample.save() 


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
    # subprocess.run("./install_faa_data.sh", check=True)

    # import master dataset and strip white space from beacon column
    planes_master = pd.read_csv(faa_master_dataset_path, index_col="MODE S CODE HEX")
    planes_master.index = planes_master.index.str.strip()

    planes_reference = pd.read_csv(
        faa_reference_dataset_path, index_col="CODE", encoding="utf-8-sig"
    )

    dataset = fo.load_dataset(voxel51_dataset_name)

    for row in dataset:
        # render plane_id in lowercase letters
        plane_icao24 = row["icao24"].label.upper()
        # find plane model code associated with the icao24 code, i.e. mode s code hex
        try:
            model_code = planes_master.loc[plane_icao24, "MFR MDL CODE"]
        except IndexError:
            logging.info(
                "Plane ID not found in master dataset. Plane ID: %s", plane_icao24
            )
            continue
        except KeyError:
            logging.info(
                "Plane ID not found in master dataset. Plane ID: %s", plane_icao24
            )
            continue
        # find reference row with all relevant model data
        plane_reference_row = planes_reference.loc[model_code]
        # exract all relevant data from plane_reference_row
        # convert all fields to string
        manufacturer = str(plane_reference_row["MFR"]).rstrip()
        model_name = str(plane_reference_row["MODEL"]).rstrip()
        aircraft_type = str(plane_reference_row["TYPE-ACFT"])
        engine_type = str(plane_reference_row["TYPE-ENG"])
        num_engines = str(plane_reference_row["NO-ENG"])
        num_seats = str(plane_reference_row["NO-SEATS"])
        aircraft_weight = str(plane_reference_row["AC-WEIGHT"])
        # norm_model = normalize_single_model_value(model_name)

        # store values in voxel51 dataset row
        row["model_code"] = fo.Classification(label=model_code)
        row["manufacturer"] = fo.Classification(label=manufacturer)
        row["model_name"] = fo.Classification(label=model_name)
        row["aircraft_type"] = fo.Classification(label=aircraft_type)
        row["engine_type"] = fo.Classification(label=engine_type)
        row["num_engines"] = fo.Classification(label=num_engines)
        row["num_seats"] = fo.Classification(label=num_seats)
        row["aircraft_weight"] = fo.Classification(label=aircraft_weight)

        # if norm_model is not None:
        #    sample["norm_model"] = fo.Classification(label=norm_model)
        row.save()

    return dataset


def normalize_model_values(dataset_name):
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

    dataset = fo.load_dataset(dataset_name)

    # json file storing plane model strings as key and standardized model
    # as value
    with open("plane_model_dict.json", "r") as file_path:
        plane_model_dict = json.load(file_path)

    # Loop thru each row of model column
    for sample in dataset.exists("model_name"):
        model = sample["model_name"].label
        norm_model = plane_model_dict.get(model, None)
        #print("{} = {}".format(model, norm_model))
        if norm_model is not None:
            sample["norm_model"] = fo.Classification(label=norm_model)
            sample.save()
        else:
            logging.info("Match not found for: %s", model)

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

def add_normalized_model_to_plane_detection(dataset_name, prediction_field, output_field):
    dataset = fo.load_dataset(dataset_name)
    for sample in dataset.exists("norm_model"):
        new_detections = sample[prediction_field].copy()

        for detection in new_detections["detections"]:
            detection["label"] = sample["norm_model"]["label"]
        
        sample[output_field] = new_detections
        sample.save()
