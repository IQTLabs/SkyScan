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
