"""Functions for training an object detection model."""

import json
import subprocess


def train_detection_model(training_name, chosen_model):
    """Train an object detection model.

    Args:
        training_name (str) - user-selected name for model

    Returns:
        ...
    """

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


def clone_tensorflow_repo():
    """Git clone tensorflow."""
    # deterministically download tensorflow v2.5.0
    # note: this command will fail on OS X catalina without further changes. Use
    # a linux machine to avoid this problem.
    command = "git clone --depth 1 https://github.com/tensorflow/models/tree/v2.5.0 /tf/models"
    subprocess.run(command.split())
