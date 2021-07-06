"""Functions for training an object detection model."""

import json
import logging
import math
import sys

import fiftyone as fo


def train_detection_model(training_name, chosen_model):
    """Train an object detection model.

    Args:
        training_name (str) - user-selected name for model

    Returns:
        ...
    """

    #TODO: cover corner case where user restarts training

    # enforce unique model name
    if os.path.isdir("/tf/ml-model/dataset-export/" + training_name):
        logging.error("Must use unique model name.")
        sys.exit(1)

    base_models = load_base_models_json()

    set_filenames(base_models, training_name, chosen_model)


def load_base_models_json(filename="base_models.json"):
    """Load base models json to allow selecting pre-trained model.

    Args:
        filename (str) - filename for the json file with pre-trained models

    Returns:
        base_models - python dict version of JSON key-value pairs
    """
    with open(filename) as json_file:
        base_models = json.load(json_file)

    return base_models


def set_filenames(base_models, training_name, chosen_model):
    """Set filename values needed for object detection training.

    Args:
        base_models (dict) - possible pre-trained models
        training_name (str) - user-selected name for model
        chosen_model (str) - the user-selected pre-trained model

    Returns:
        filepaths (dict) - keys are names, values are filenames
    """
    filepaths = {}

    # intermediate variables needed later for filename construction
    base_pipeline_file = base_models[chosen_model]["base_pipeline_file"]
    model_name = base_models[chosen_model]["model_name"]

    # set all filepath key-value pairs
    filepaths["train_record_file"] = (
        "/tf/dataset-export/" + training_name + "/train/tf.records"
    )
    filepaths["val_record_file"] = (
        "/tf/dataset-export/" + training_name + "/val/tf.records"
    )
    filepaths["val_export_dir"] = "/tf/dataset-export/" + training_name + "/val/"
    filepaths["train_export_dir"] = "/tf/dataset-export/" + training_name + "/train/"
    filepaths["model_export_dir"] = "/tf/model-export/" + training_name + "/"
    filepaths["label_map_file"] = (
        "/tf/dataset-export/" + training_name + "/label_map.pbtxt"
    )
    filepaths["model_dir"] = "/tf/training/" + training_name + "/"
    filepaths["pretrained_checkpoint"] = base_models[chosen_model][
        "pretrained_checkpoint"
    ]
    filepaths["pipeline_file"] = "/tf/models/research/deploy/" + base_pipeline_file
    filepaths["fine_tune_checkpoint"] = (
        "/tf/models/research/deploy/" + model_name + "/checkpoint/ckpt-0"
    )
    # TODO: Return to this later. Is this a bug or not? Will find out later when
    # I get further into this module.
    # filepaths["pipeline_file"] = "/tf/models/research/deploy/pipeline_file.config"

    return filepaths


def export_voxel51_dataset_to_tfrecords(
    dataset_name, filepaths, label_field, training_percentage=0.8
):
    """Export the voxel51 dataset to TensorFlow records.

    Args:
        dataset_name (str) - voxel51 dataset name
        filepaths (dict) - filename values created by set_filenames
        label_field (str) - label field set in config
        training_percentage (float) - percentage of sample for training

    Returns:
        None
    """
    # load voxel51 dataset and create a view
    dataset = fo.load_dataset(dataset_name)
    view = dataset.match_tags("training").shuffle(seed=51)

    # calculate size of training and validation set
    sample_len = len(view)
    train_len = math.floor(sample_len * training_percentage)
    val_len = math.floor(sample_len * (1 - training_percentage))

    # extract training and validation records
    val_view = view.take(val_len)
    train_view = view.skip(val_len).take(train_len)

    # Export the validation and training datasets
    val_view.export(
        export_dir=filepaths["val_export_dir"],
        dataset_type=fo.types.TFObjectDetectionDataset,
        label_field=label_field,
    )
    train_view.export(
        export_dir=filepaths["train_export_dir"],
        dataset_type=fo.types.TFObjectDetectionDataset,
        label_field=label_field,
    )
