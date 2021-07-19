"""Utility functions related to labelbox."""

import logging
import sys

import fiftyone as fo
import fiftyone.utils.labelbox as foul
from labelbox import Client, Project, Dataset

# pylint: disable=C0330, R0913, E1205


def upload_vox51_dataset_to_labelbox(
    labelbox_api_key,
    labelbox_dataset_name,
    labelbox_project_name,
    voxel51_dataset_name,
    labelbox_id_field="labelbox_id",
    num_samples=500,
):
    """Upload a voxel51 dataset to labelbox.

    Args:
        labelbox_api_key (str)
        labelbox_dataset_name (str)
        labelbox_project_name (str)
        voxel51_dataset_name (str)
        lablebox_id_field (str) - unique ID required for upload of dataset
        num_samples - number of images to randomly choose for upload

    Returns:
        None
    """
    # TODO: Some sort of problem related to labelbox ID
    logging.info("Uploading voxel51 dataset to Labelbox.")

    # set up voxel51 and labelbox connections
    dataset = fo.load_dataset(voxel51_dataset_name)
    client = Client(labelbox_api_key)
    # must convert PaginatedCollection to list in order to count length
    projects = list(client.get_projects(where=Project.name == labelbox_project_name))

    # ensure there is only labelbox project of specified name
    num_labelbox_projects = len(projects)
    if num_labelbox_projects != 1:
        logging.error(
            "Expected a single project named: %s but found %s projects",
            labelbox_project_name,
            num_labelbox_projects,
        )
        sys.exit(1)

    project = list(projects)[0]

    # select proper labelbox dataset
    # must convert PaginatedCollection to list in order to count length
    labelbox_datasets = list(project.datasets(where=Dataset.name == labelbox_dataset_name))

    # ensure there is only one labelbox dataset of specified name
    num_labelbox_datasets = len(labelbox_datasets)
    if num_labelbox_datasets != 1:
        logging.info(
            "Expected a single dataset named: {} but found {} projects",
            labelbox_dataset_name,
            num_labelbox_datasets,
        )
        sys.exit(1)

    labelbox_dataset = list(labelbox_datasets)[0]

    # take random sample of images and upload to labelbox
    view = dataset.shuffle().take(num_samples)

    # add a "training" tag to all of the samples being sent to labelbox
    for sample in view:
        sample.tags.append("training")
        sample.save()

    foul.upload_media_to_labelbox(labelbox_dataset, view, labelbox_id_field)

def resume_upload_vox51_dataset_to_labelbox(
    labelbox_api_key,
    labelbox_dataset_name,
    labelbox_project_name,
    voxel51_dataset_name,
    labelbox_id_field="labelbox_id",
    num_samples=500,
):
    """Upload a voxel51 dataset to labelbox.

    Args:
        labelbox_api_key (str)
        labelbox_dataset_name (str)
        labelbox_project_name (str)
        voxel51_dataset_name (str)
        lablebox_id_field (str) - unique ID required for upload of dataset
        num_samples - number of images to randomly choose for upload

    Returns:
        None
    """
    # TODO: Some sort of problem related to labelbox ID
    logging.info("Uploading voxel51 dataset to Labelbox.")

    # set up voxel51 and labelbox connections
    dataset = fo.load_dataset(voxel51_dataset_name)
    client = Client(labelbox_api_key)
    # must convert PaginatedCollection to list in order to count length
    projects = list(client.get_projects(where=Project.name == labelbox_project_name))

    # ensure there is only labelbox project of specified name
    num_labelbox_projects = len(projects)
    if num_labelbox_projects != 1:
        logging.error(
            "Expected a single project named: %s but found %s projects",
            labelbox_project_name,
            num_labelbox_projects,
        )
        sys.exit(1)

    project = list(projects)[0]

    # select proper labelbox dataset
    # must convert PaginatedCollection to list in order to count length
    labelbox_datasets = list(project.datasets(where=Dataset.name == labelbox_dataset_name))

    # ensure there is only one labelbox dataset of specified name
    num_labelbox_datasets = len(labelbox_datasets)
    if num_labelbox_datasets != 1:
        logging.info(
            "Expected a single dataset named: {} but found {} projects",
            labelbox_dataset_name,
            num_labelbox_datasets,
        )
        sys.exit(1)

    labelbox_dataset = list(labelbox_datasets)[0]

    # take random sample of images and upload to labelbox
    view = dataset.match_tags("training")

    foul.upload_media_to_labelbox(labelbox_dataset, view, labelbox_id_field)


def merge_labelbox_dataset_with_voxel51(
    voxel51_dataset_name, labelbox_json_path, labelbox_id_field="labelbox_id"
):
    """Merge the labels created via labelbox with the voxel51 dataset.

    The json referenced in the labelbox_json_path must be manually downloaded
    from the labelbox website.

    Args:
        voxel51_dataset_name (str)
        labelbox_json_path (str) - a path to
        lablebox_id_field (str) - unique ID required for merging of dataset

    Returns:
        None
    """
    dataset = fo.load_dataset(voxel51_dataset_name)

    foul.import_from_labelbox(
        dataset, labelbox_json_path, labelbox_id_field=labelbox_id_field
    )