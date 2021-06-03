"""custom functionality related to voxel51 databases"""

import os

import pandas as pd
import fiftyone as fo


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
        print("Created {} dataset".format(dataset_name))
    # If the dataset already exists, load it instead
    except ValueError:
        dataset = fo.load_dataset(name=dataset_name)
        print("Dataset already exists. Loaded {} dataset".format(dataset_name))

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
                print(plane.size)
        except KeyError:
            print("FAA Data entry not found for: {}".format(plane_id))

    return dataset

def normalize_models():
    # TODO: Fill out this function
    pass

    # models = []
    # view = dataset.exists("model")

    # for s in view.select_fields("model"):
    #     if s.model.label not in models:
    #         models.append(s.model.label)
    # for model in models:
    #     print(model)

    # TODO: probably convert this into a json dict structure that can then be more easily
    # understood within the code
    # 
    # def normalizeModel(model):
    # norm_model=None
    # if model=="A319-112" or model=="A319-115" or model=="A319-132":
    #     norm_model="A319"

    # if model=="A320 232" or model=="A320 232SL" or model=="A320-211" or model=="A320-212" or model=="A320-214" or model=="A320-232" or model=="A320-251N" or model=="A320-271N":
    #     norm_model="A320"

    # if model=="A321-211" or model=="A321-231" or model=="A321-271NX" or model=="Airbus A321-231(SL)":
    #     norm_model="A321"

    # if model=="A330 243" or model=="A330-243F":
    #     norm_model="A330"

    # if model=="737-71B" or model=="737-724" or model=="737-73V" or model=="737-752" or model=="737-7H4" or model=="737-7Q8":
    #     norm_model="737-700"

    # if model=="737-800" or model=="Boeing 737-852" or model=="737-823" or model=="737-824" or model=="737-832" or model=="737-83N" or model=="737-84P" or model=="737-890" or model=="737-8EH" or model=="737-8H4" or model=="737NG 823/W" or model=="737NG 852/W" or model=="737NG 85P/W" or model=="737NG 86N/W" or model=="737NG 8V3/W":
    #     norm_model="737-800"

    # if model=="737-900ER" or model=="737-924ER" or model=="737-932ER":
    #     norm_model="737-900"

    # if model=="747-48EF":
    #     norm_model="747-400"

    # if model=="757-231" or model=="757-232" or model=="757-251":
    #     norm_model="757-200"

    # if model=="767 330ER/W":
    #     norm_model="767-300"

    # if model=="777-223":
    #     norm_model="777-200"

    # if model=="787-8":
    #     norm_model="787-800"

    # if model=="787-9 (Boeing)" or model=="BOEING 787-9 Dreamliner":
    #     norm_model="787-800"

    # if model=="45" or model=="60":
    #     norm_model="Learjet 45/60"

    # if model=="510" or model=="Citation Excel" or model=="Citation Sovereign+" or model=="525" or model=="550" or model=="560" or model=="680" or model=="750" or model=="525A" or model=="525B" or model=="525C" or model=="560XL" or model=="680A":
    #     norm_model="Cessna Jet"

    # if model=="CL-600-2B16" or model=="BD-100-1A10" or model=="BD-700-1A11":
    #     norm_model="Bombardier Challanger"

    # if model=="CL-600-2C10":
    #     norm_model="CRJ700"

    # if model=="CL-600-2C11":
    #     norm_model="CRJ550"

    # if model=="CL-600-2D24" or model=="CRJ 900 LR NG" or model=="CRJ-900":
    #     norm_model="CRJ900"

    # if model=="ERJ 170-100 SE" or model=="ERJ 170-100SU" or model=="ERJ 170-200 LR" or model=="ERJ 190-100 IGW" or model=="EMB-190 AR":
    #     norm_model="ERJ-170"

    # if model=="EMB-135BJ" or model=="EMB-145LR":
    #     norm_model="EMB-135"

    # if model=="EMB-505" or model=="EMB-545":
    #     norm_model="EMB-505"

    # if model=="PA-23-250":
    #     norm_model="Piper PA-23"

    # if model=="PC-12/47E":
    #     norm_model="Pilatus PC-12"
        
    # if model=="FALCON 10" or model=="FALCON 2000" or model=="FALCON 50" or model=="FALCON 7X" or model=="FALCON 900 EX" or model=="FALCON 900EX":    
    #     norm_model="Falcon"
    
    # if model=="G-IV" or model=="G-V" or model=="GALAXY" or model=="GIV-X (G450)" or model=="GULFSTREAM 200" or model=="GV-SP (G550)" or model=="GVI(G650ER)":
    #     norm_model="Gulfstream"
        
    # if model=="HAWKER 800XP" or model=="HAWKER 900XP":
    #     norm_model="Hawker"
    
    # if model=="SF50":
    #     norm_model="Cirrus"
    
    # if model=="PRESSURIZED LANCR IV":
    #     norm_model="Lancair IV"
        
    # if model=="B300":
    #     norm_model="King Air"

    # return norm_model

    # Finally, actually apply normalization
    # for sample in dataset.exists("model"):
    #     norm_model = normalizeModel(sample["model"].label)
    #     if norm_model != None:
    #         sample["norm_model"] = fo.Classification(label=norm_model)
    #         sample.save()
    #     else:
    #         print("Match not found for: {}".format(sample["model"].label))

    #     #addPlaneData(sample)
