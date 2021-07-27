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
    normalize_model_values,
    build_multi_class_train_eval_dataset,
    add_normalized_model_to_plane_detection
)

from labelbox_utils import (
    upload_vox51_dataset_to_labelbox,
    merge_labelbox_dataset_with_voxel51,
)

from detection import export_detection_model, train_detection_model

from prediction import (run_detection_model, run_detection_model_tiled)

from evaluation import evaluate_detection_model

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
    config["DEFAULT"] = {
        "num_eval_steps": 500,
        "label_field": "detections",
        "tile_string": "1920x1080,768x768",
        "tile_overlap": 50,
        "iou_threshold": 0,
        "upload_num_samples": 500,
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
        "--upload_train",
        default=False,  # default value is False
        action="store_true",
        help="Upload train samples to labelbox.",
    )
    parser.add_argument(
        "--resume_upload_train",
        default=False,  # default value is False
        action="store_true",
        help="Resume uploading train samples to labelbox.",
    )
    parser.add_argument(
        "--upload_eval",
        default=False,  # default value is False
        action="store_true",
        help="Upload eval samples to labelbox.",
    )
    parser.add_argument(
        "--resume_upload_eval",
        default=False,  # default value is False
        action="store_true",
        help="Resume uploading eval samples to labelbox.",
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
        "--train_multi_class",
        default=False,  # default value is False
        action="store_true",
        help="Train a multi-class model.",
    )

    parser.add_argument(
        "--predict",
        default=False,  # default value is False
        action="store_true",
        help="Model prediction.",
    )

    parser.add_argument(
        "--evaluate",
        default=False,  # default value is False
        action="store_true",
        help="Model evaluation.",
    )

    parser.add_argument(
        "--normalize",
        default=False,  # default value is False
        action="store_true",
        help="Normalize plane data",
    )

    parser.add_argument(
        "--predict_tiled",
        default=False,  # default value is False
        action="store_true",
        help="Tiled model prediction.",
    )

    parser.add_argument(
        "--build_multi_class_dataset",
        default=False,  # default value is False
        action="store_true",
        help="Build multi-class dataset.",
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

    # check if user selected data normalization stage
    if args.normalize:
        # check that config file contains both dataset name
        if (
            config["file_names"]["dataset_name"]
        ):
            logging.info("Entering 'normalize data' route.")

            # we should be able to remove this function - I had to do it
            # after the fact this time because some of the files were 
            # imported via the Jupyter script
            #dataset_with_faa_data = add_faa_data_to_voxel51_dataset(
            #    config["file_names"]["dataset_name"],
            #    "../data/faa_master.txt",
            #    "../data/faa_aircraft_reference.txt",
            #)


            normalize_model_values(config["file_names"]["dataset_name"])
            
            logging.info("Exiting 'normalize data' route.")
        # exit if config file does not contain the dataset name.
        else:
            logging.info(
                "Missing config file value image for the dataset_name."
            )
            sys.exit(1)  # exit program



    # check if user selected upload train to labelbox stage
    if args.upload_train:
        if all(
            [
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
            ]
        ):
            logging.info("Entering 'upload train samples to labelbox' route.")
            upload_vox51_dataset_to_labelbox(
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
                config.getint("upload","upload_num_samples"),
                "train",
                "eval",
                False
            )
            logging.info("Exiting 'upload train samples to labelbox' route.")
        else:
            logging.info(
                """Missing config file value for labelbox API key, lablebox dataset name,
                labelbox project name or voxel51 dataset name."""
            )
            sys.exit(1)  # exit program

    # check if user selected upload eval to labelbox stage
    if args.upload_eval:
        if all(
            [
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
            ]
        ):
            logging.info("Entering 'upload eval samples to labelbox' route.")
            upload_vox51_dataset_to_labelbox(
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
                config.getint("upload","upload_num_samples"),
                "eval", 
                "train",
                False
            )
            logging.info("Exiting 'upload eval samples to labelbox' route.")
        else:
            logging.info(
                """Missing config file value for labelbox API key, lablebox dataset name,
                labelbox project name or voxel51 dataset name."""
            )
            sys.exit(1)  # exit program

    # check if user selected resume_upload_train to labelbox stage
    if args.resume_upload_train:
        if all(
            [
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
            ]
        ):
            logging.info("Entering 'resume uploading train samples to labelbox' route.")
            upload_vox51_dataset_to_labelbox(
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
                config.getint("upload","upload_num_samples"),
                "train",
                "eval",
                resume=True
            )
            logging.info("Exiting 'resume uploading train samples to labelbox' route.")
        else:
            logging.info(
                """Missing config file value for labelbox API key, lablebox dataset name,
                labelbox project name or voxel51 dataset name."""
            )
            sys.exit(1)  # exit program

    # check if user selected resume_upload_eval to labelbox stage
    if args.resume_upload_eval:
        if all(
            [
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
            ]
        ):
            logging.info("Entering 'resume upload dataset to labelbox' route.")
            upload_vox51_dataset_to_labelbox(
                config["labelbox"]["api_key"],
                config["labelbox"]["dataset_name"],
                config["labelbox"]["project_name"],
                config["file_names"]["dataset_name"],
                config.getint("upload","upload_num_samples"),
                "train",
                "eval",
                resume=False
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

    # check if user selected build multi-class dataset stage
    if args.build_multi_class_dataset:
        if (
            config["file_names"]["dataset_name"],
            config["prediction"]["prediction_field"]
        ):
            logging.info("Entering 'build multi-class dataset' route.")
            #add_normalized_model_to_plane_detection(config["file_names"]["dataset_name"], config["prediction"]["prediction_field"], "multi_class_detections")
            build_multi_class_train_eval_dataset(
                config["file_names"]["dataset_name"]
            )
            logging.info("Exiting 'build multi-class dataset' route.")
        else:
            logging.info(
                """Missing config file value for voxel51 dataset name."""
            )
            sys.exit(1)  # exit program


    # check if user selected train model stage
    if args.train:
        if all(
            [
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["model"]["base_model"],
                config["model"]["num_train_steps"],
            ]
        ):
            logging.info("Entering 'train model' route.")
            train_detection_model(
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["model"]["base_model"],
                config.getint("model","num_train_steps"),
                label_field="detections"
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


    # check if user selected train multi-class model stage
    if args.train_multi_class:
        if all(
            [
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["model"]["base_model"],
                config["model"]["num_train_steps"],
            ]
        ):
            logging.info("Entering 'train multi-class model' route.")
            train_detection_model(
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["model"]["base_model"],
                config.getint("model","num_train_steps"),
                label_field = "multi_class_detections",
                training_tag = "multi_class_train"
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
        if all(
            [
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["model"]["base_model"],
            ]
        ):
            logging.info("Entering 'export model' route.")
            export_detection_model(
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["model"]["base_model"],
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

    # check if user selected model prediction stage
    if args.predict:
        if all(
            [
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["prediction"]["prediction_field"],
            ]
        ):
            logging.info("Entering 'model prediction' route.")
            run_detection_model(
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["prediction"]["prediction_field"],
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
        if all(
            [
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["prediction"]["prediction_field"],
            ]
        ):
            logging.info("Entering 'model prediction tiled' route.")
            run_detection_model_tiled(
                config["file_names"]["dataset_name"],
                config["model"]["training_name"],
                config["prediction"]["prediction_field"],
                config["prediction"]["tile_string"],
                config.getint("prediction","tile_overlap"),
                config.getfloat("prediction","iou_threshold"),
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

    # check if user selected evaluate model stage
    if args.evaluate:
        if all(
            [
                config["file_names"]["dataset_name"],
                config["prediction"]["prediction_field"],
                config["evaluation"]["evaluation_key"],
            ]
        ):
            logging.info("Entering 'model evaluation' route.")
            evaluate_detection_model(
                config["file_names"]["dataset_name"],
                config["prediction"]["prediction_field"],
                config["evaluation"]["evaluation_key"],
            )
            logging.info("Exiting 'model evaluation' route.")
        else:
            logging.info(
                """Missing one or more config file values required for evaluation:
                - file_names / dataset_name
                - prediction / prediction_field
                - evaluation / evaluation_key"""
            )
            sys.exit(1)  # exit program
