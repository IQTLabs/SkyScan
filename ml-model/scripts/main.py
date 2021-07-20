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
    resume_upload_vox51_dataset_to_labelbox,
    upload_vox51_dataset_to_labelbox,
    merge_labelbox_dataset_with_voxel51,
)

from detection import (
    export_detection_model,
    train_detection_model
)

from prediction import (
    run_detection_model
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
    config['DEFAULT'] = {
        "num_eval_steps": 500,
        "label_field": "detections",
        "tile_string": "1920x1080,768x768",
        "tile_overlap": 50,
        "iou_threshold": 0,
        "upload_tag": "training",
        "num_upload_samples": 500
    }
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
        "--resume_upload",
        default=False,  # default value is False
        action="store_true",
        help="Resume upload dataset to labelbox.",
    )
    parser.add_argument(
        "--download",
        "--download_from_labelbox",
        default=False,  # default value is False
        action="store_true",
        help="Download dataset from labelbox.",
    )

    parser.add_argument(
        "--train",
        default=False,  # default value is False
        action="store_true",
        help="Train a model.",
    )

    parser.add_argument(
        "--predict",
        default=False,  # default value is False
        action="store_true",
        help="Model prediction.",
    )

    parser.add_argument(
        "--predict_tiled",
        default=False,  # default value is False
        action="store_true",
        help="Tiled model prediction.",
    )

    parser.add_argument(
        "--export_model",
        default=False,  # default value is False
        action="store_true",
        help="Export a trained model.",
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
                config["upload"]["upload_num_samples"],
                config["upload"]["upload_tag"]
            )
            logging.info("Exiting 'upload dataset to labelbox' route.")
        else:
            logging.info(
                """Missing config file value for labelbox API key, lablebox dataset name,
                labelbox project name or voxel51 dataset name."""
            )
            sys.exit(1)  # exit program

    # check if user selected resume_upload to labelbox stage
    if args.resume_upload:
        if all(
            [
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
            ]
        ):
            logging.info("Entering 'resume upload dataset to labelbox' route.")
            resume_upload_vox51_dataset_to_labelbox(
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
                config["upload"]["upload_tag"]
            )
            logging.info("Exiting 'resume upload dataset to labelbox' route.")
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


    # check if user selected model prediction stage
    if args.predict:
        if all ([
            config["file_names"]["dataset_name"],
            config["model"]["training_name"],
            config["prediction"]["prediction_field"]
        ]
        ):
            logging.info("Entering 'model prediction' route.")
            run_detection_model(
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["prediction"]["prediction_field"]
            )
            logging.info("Exiting 'model prediction' route.")
        else:
            logging.info(
                """Missing one or more config file values required for prediction:
                - file_names / dataset_name
                - model / training_name
                - prediction / prediction_field"""
            )
            sys.exit(1)  # exit program


    # check if user selected model prediction tiled stage
    if args.predict_tiled:
        if all ([
            config["file_names"]["dataset_name"],
            config["model"]["training_name"],
            config["prediction"]["prediction_field"]
        ]
        ):
            logging.info("Entering 'model prediction tiled' route.")
            run_detection_model(
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["prediction"]["prediction_field"],
                config["prediction"]["tile_string"],
                config["prediction"]["tile_overlap"],
                config["prediction"]["iou_threshold"]
            )
            logging.info("Exiting 'model prediction tiled' route.")
        else:
            logging.info(
                """Missing one or more config file values required for prediction tiled:
                - file_names / dataset_name
                - model / training_name
                - prediction / prediction_field"""
            )
            sys.exit(1)  # exit program





    # check if user selected train model stage
    if args.train:
        if all ([
            config["file_names"]["dataset_name"],
            config["model"]["training_name"],
            config["model"]["base_model"],
            config["model"]["num_train_steps"]
        ]
        ):
            logging.info("Entering 'train model' route.")
            train_detection_model(
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["model"]["base_model"],
                config["model"]["num_train_steps"],
                config["model"]["label_field"],
                config["model"]["num_eval_steps"]
            )
            logging.info("Exiting 'train model' route.")
        else:
            logging.info(
                """Missing one or more config file values required for training:
                - file_names / dataset_name
                - model / training_name
                - model / base_model
                - model / num_train_steps"""
            )
            sys.exit(1)  # exit program

    # check if user selected export model stage
    if args.export_model:
        if all ([
            config["file_names"]["dataset_name"],
            config["model"]["training_name"],
            config["model"]["base_model"]
        ]
        ):
            logging.info("Entering 'export model' route.")
            export_detection_model(
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["model"]["base_model"]
            )
            logging.info("Exiting 'export model' route.")
        else:
            logging.info(
                """Missing one or more config file values required for exporting a model:
                - file_names / dataset_name
                - model / training_name
                - model / base_model"""
            )
            sys.exit(1)  # exit program
