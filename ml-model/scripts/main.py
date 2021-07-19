"""Control logic for running skyscan data and model functionality."""

import argparse
import configparser
import logging
import os
import sys

from customvox51 import (
    add_faa_data_to_voxel51_dataset,
    add_sample_images_to_voxel51_dataset,
    build_image_list,
    create_voxel51_dataset,
)

from labelbox_utils import (
    upload_vox51_dataset_to_labelbox,
    merge_labelbox_dataset_with_voxel51,
)

# pylint: disable=C0330, W0621


def read_config(config_file=os.path.join("config", "config.ini")):
    """Read in config file values.

    Use python standard library configparser module
    to read in user-defined values. This enables configurability.

    For more info on configparser, see here:
    https://docs.python.org/3/library/configparser.html#module-configparser

    Args:
        config file (str) - name of config file. Default is config.ini

    Returns:
        config - a config object similar to a dict
    """

    logging.info("Starting to read config file.")
    config = configparser.ConfigParser()
    config.read(config_file)
    logging.info("Finished reading config file.")
    return config


def parse_command_line_arguments():
    """Parse command line arguments with argparse."""
    parser = argparse.ArgumentParser(
        description="Run skyscan data and model scripts.",
        epilog="For help with this program, contact John Speed at jmeyers@iqt.org.",
    )
    parser.add_argument(
        "--prep",
        default=False,  # default value is False
        action="store_true",
        help="Prepare voxel51 dataset.",
    )
    parser.add_argument(
        "--upload",
        "--upload_to_labelbox",
        default=False,  # default value is False
        action="store_true",
        help="Upload dataset to labelbox.",
    )
    parser.add_argument(
        "--download",
        "--download_from_labelbox",
        default=False,  # default value is False
        action="store_true",
        help="Download dataset from labelbox.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_arguments()
    config = read_config()

    # check if user selected data preparation stage
    if args.prep:
        # check that config file contains both dataset name and
        # image_directory
        if (
            config["file_names"]["dataset_name"]
            and config["file_locations"]["image_directory"]
        ):
            logging.info("Entering 'prepare data' route.")
            image_list = build_image_list(config["file_locations"]["image_directory"])
            dataset = create_voxel51_dataset(config["file_names"]["dataset_name"])
            modified_dataset = add_sample_images_to_voxel51_dataset(
                image_list, dataset, config["import"]["datasource_name"]
            )
            dataset_with_faa_data = add_faa_data_to_voxel51_dataset(
                config["file_names"]["dataset_name"],
                "../data/faa_master.txt",
                "../data/faa_aircraft_reference.txt",
            )
            logging.info("Exiting 'prepare data' route.")
        # exit if config file does not contain image directory or dataset name.
        else:
            logging.info(
                "Missing config file value image for image directory or dataset_name."
            )
            sys.exit(1)  # exit program

    # check if user selected upload to labelbox stage
    if args.upload:
        if all(
            [
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
            ]
        ):
            logging.info("Entering 'upload dataset to labelbox' route.")
            upload_vox51_dataset_to_labelbox(
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
            )
            logging.info("Exiting 'upload dataset to labelbox' route.")
        else:
            logging.info(
                """Missing config file value for labelbox API key, lablebox dataset name,
                labelbox project name or voxel51 dataset name."""
            )
            sys.exit(1)  # exit program

    # check if user selected download from labelbox stage
    if args.download:
        if (
            config["file_names"]["dataset_name"]
            and config["labelbox"]["exported_json_path"]
        ):
            logging.info("Entering 'download from labelbox' route.")
            merge_labelbox_dataset_with_voxel51(
                config["file_names"]["dataset_name"],
                config["labelbox"]["exported_json_path"],
            )
            logging.info("Exiting 'download from labelbox' route.")
        else:
            logging.info(
                """Missing config file value for voxel51 dataset name and
                labelbox exported JSON path."""
            )
            sys.exit(1)  # exit program
