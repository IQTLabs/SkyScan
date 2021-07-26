"""Functions for training an object detection model."""

import json
import logging
import math
import os
import os.path
import re
import subprocess
import sys
import tarfile
import shutil

import fiftyone as fo
from google.protobuf import text_format
from object_detection.protos.string_int_label_map_pb2 import (
    StringIntLabelMap,
    StringIntLabelMapItem,
)
from object_detection.utils import label_map_util


def export_detection_model(dataset_name, training_name, chosen_model):
    base_models = load_base_models_json()

    filepaths = set_filenames(base_models, training_name, chosen_model)

    logging.info(
        "Exporting detection model to: {}".format(
            filepaths["image_tensor_model_export_dir"]
        )
    )

    """Call model to initiate training."""
    pipeline_file = filepaths["pipeline_file"]
    model_dir = filepaths["model_dir"]
    image_tensor_model_export_dir = filepaths["image_tensor_model_export_dir"]

    command = """python /tf/models/research/object_detection/exporter_main_v2.py \
    --input_type image_tensor \
    --trained_checkpoint_dir={model_dir} \
    --pipeline_config_path={pipeline_file} \
    --output_directory {image_tensor_model_export_dir}""".format(
        model_dir=model_dir,
        pipeline_file=pipeline_file,
        image_tensor_model_export_dir=image_tensor_model_export_dir,
    )

    subprocess.run(command.split(), check=True)

    # copy the label_map file into the model export directory
    shutil.copyfile(filepaths["label_map_file"], filepaths["model_export_dir"] + "label_map.pbtxt")

def train_detection_model(
    dataset_name,
    training_name,
    chosen_model,
    num_train_steps,
    label_field="detections",
    num_eval_steps:int=500,
):
    """Train an object detection model.

    Args:
        training_name (str) - user-selected name for model

    Returns:
        ...
    """

    # TODO: cover corner case where user restarts training

    # enforce unique model name
    if os.path.isdir("/tf/ml-model/dataset-export/" + training_name):
        logging.error("Must use unique model name.")
        sys.exit(1)

    base_models = load_base_models_json()

    filepaths = set_filenames(base_models, training_name, chosen_model)

    export_voxel51_dataset_to_tfrecords(dataset_name, filepaths, label_field)

    detection_mapping = create_detection_mapping(dataset_name, label_field)

    save_mapping_to_file(detection_mapping, filepaths)

    download_base_training_config(filepaths)

    download_pretrained_model(filepaths)

    create_custom_training_config_file(
        base_models, chosen_model, filepaths, num_train_steps
    )

    call_train_model(filepaths, num_train_steps, num_eval_steps)


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
    filepaths["image_tensor_model_export_dir"] = (
        "/tf/model-export/" + training_name + "/image_tensor_saved_model/"
    )
    filepaths["label_map_file"] = (
        "/tf/dataset-export/" + training_name + "/label_map.pbtxt"
    )
    filepaths["model_dir"] = "/tf/training/" + training_name + "/"
    filepaths["pretrained_checkpoint"] = base_models[chosen_model][
        "pretrained_checkpoint"
    ]
    filepaths["fine_tune_checkpoint"] = (
        "/tf/models/research/deploy/" + model_name + "/checkpoint/ckpt-0"
    )
    filepaths["base_pipeline_file"] = base_pipeline_file
    filepaths["base_pipeline_dir"] = "/tf/models/research/deploy/"
    filepaths["pipeline_file"] = (
        "/tf/dataset-export/" + training_name + "/pipeline_file.config"
    )

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

    if os.path.isfile(filepaths["val_export_dir"] + "tf.records"):
        logging.info("TF Records already exist, skipping export.")
        return
    # load voxel51 dataset and create a view
    dataset = fo.load_dataset(dataset_name)
    view = dataset.match_tags("train").shuffle(seed=51)

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


def create_detection_mapping(dataset_name, label_field):
    """Create mapping from labels to IDs.

    This function creates a mapping necessary for tensorflow
    object detection.

    Args:
        dataset_name (str) - voxel51 dataset name
        label_field (str) - label field set in config

    Returns:
        mapping (str) - mapping of labels to IDs
    """
    # pylint: disable=invalid-name,no-member,redefined-builtin

    logging.info("Creating detection classes to ID mapping.")

    # load voxel51 dataset and create a view
    dataset = fo.load_dataset(dataset_name)
    view = dataset.match_tags("train").shuffle(seed=2021)

    # create a list of all class names
    class_names = _create_list_of_class_names(view, label_field)

    # convert list of class names to a mapping data structure
    # this mapping data structure uses a name as one field and a unique
    # integer id in the other field. It helps the model map a string label
    # name to an id number.
    # for detailed info, see here:
    # https://github.com/tensorflow/models/blob/master/research/object_detection/protos/string_int_label_map.proto
    msg = StringIntLabelMap()
    for id, name in enumerate(class_names, start=1):  # start counting at 1, not 0
        msg.item.append(StringIntLabelMapItem(id=id, name=name))
    mapping = str(text_format.MessageToBytes(msg, as_utf8=True), "utf-8")
    logging.info("Finished creating detection classes to ID mapping.")

    return mapping


def _create_list_of_class_names(view, label_field):
    """Create list of class names from the label field.

    Args:
        view (voxel51 view object) - the voxel51 dataset
        label_field (str) - label field set in config

    Returns:
        class_names (list)
    """
    logging.info("Extracting class names from label field.")
    class_names = []
    for sample in view.select_fields(label_field):
        if sample[label_field] is not None:
            for detection in sample[label_field].detections:
                label = detection["label"]
                if label not in class_names:
                    class_names.append(label)
    logging.info("Finished extracting class names from label field.")
    return class_names


def save_mapping_to_file(mapping, filepaths):
    """Save detection classes to ID mapping file.

    Args:
        mapping - the mapping to save
        filepaths (dict) - filename values created by set_filenames

    """
    logging.info("Creating detection classes to ID mapping file.")
    with open(filepaths["label_map_file"], "w") as f:
        f.write(mapping)
    logging.info("Finished creating detection classes to ID mapping file.")


def get_num_classes_from_label_map(filepaths):
    """Retrieve number of classes from label map file.

    Args:
        mapping_filename (str)

    Returns:
        num_classes (int)
    """
    logging.info("Calculating number of classes in label map file.")
    label_map = label_map_util.load_labelmap(filepaths["label_map_file"])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)
    num_classes = len(category_index.keys())
    logging.info("Finished calculating number of classes in label map file.")
    return num_classes


def download_pretrained_model(filepaths):
    """Download pretrained machine learning model."""

    if os.path.isfile(
        "/tf/models/research/deploy/" + filepaths["pretrained_checkpoint"]
    ):
        logging.info("Pretrained model already downloaded.")
        return

    logging.info("Downloading pretrained model.")
    # specify url and download model .tar
    download_tar = (
        "http://download.tensorflow.org/models/object_detection/tf2/20200711/"
        + filepaths["pretrained_checkpoint"]
    )

    subprocess.run(
        "./install_pretrained_model.sh {}".format(download_tar).split(), check=True
    )

    # open and extract tarfile
    tar_filepath = "/tf/models/research/deploy/" + filepaths["pretrained_checkpoint"]
    with tarfile.open(tar_filepath) as tar:
        tar.extractall(path="/tf/models/research/deploy/")

    logging.info("Finished downloading pretrained model.")


def download_base_training_config(filepaths):
    """Download base training configuration file.

    Args:
        filepaths (dict) - filename values created by set_filenames

    Returns:
        None
    """

    if os.path.isfile(filepaths["base_pipeline_dir"] + filepaths["base_pipeline_file"]):
        logging.info(
            "Base training configuration file already exists, skipping download."
        )
        return

    # pylint: disable=line-too-long
    logging.info("Downloading base training configuration file.")
    # specify configuration file URL
    config_file_url = (
        "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/"
        + filepaths["base_pipeline_file"]
    )

    # run bash script to keep using same commands as jupyter notebook taken
    # from Google. This bash script could be implemented in Python.

    # TODO: capture the return value from the bash script. If the wget command
    # fails then abort the script. Or add an asert that kills the script if
    # the excpected file is not there.
    subprocess.run(
        "./install_base_training_config.sh {}".format(config_file_url).split(),
        check=True,
    )

    logging.info("Finished downloading base training configuration file.")


def create_custom_training_config_file(
    base_models, chosen_model, filepaths, num_train_steps
):
    """Download base training configuration file.

    Args:
        base_models (dict) - possible pre-trained models
        filepaths (dict) - filename values created by
        num_classes (int) - number of classes on which to do object detection

    Returns:
        None
    """
    num_classes = get_num_classes_from_label_map(filepaths)

    # pylint: disable=anomalous-backslash-in-string,line-too-long,invalid-name
    logging.info("writing custom configuration file")

    with open(filepaths["base_pipeline_dir"] + filepaths["base_pipeline_file"]) as f:
        s = f.read()

    with open(filepaths["pipeline_file"], "w") as f:

        # fine_tune_checkpoint
        s = re.sub(
            'fine_tune_checkpoint: ".*?"',
            'fine_tune_checkpoint: "{}"'.format(filepaths["fine_tune_checkpoint"]),
            s,
        )

        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")',
            'input_path: "{}"'.format(filepaths["train_record_file"]),
            s,
        )
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")',
            'input_path: "{}"'.format(filepaths["val_record_file"]),
            s,
        )

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"',
            'label_map_path: "{}"'.format(filepaths["label_map_file"]),
            s,
        )

        # Set training batch_size.
        s = re.sub(
            "batch_size: [0-9]+",
            "batch_size: {}".format(base_models[chosen_model]["batch_size"]),
            s,
        )

        # Set training steps, num_steps
        s = re.sub("num_steps: [0-9]+", "num_steps: {}".format(num_train_steps), s)

        # Set learning_rate_base in learning_rate, sane default
        #     s = re.sub('learning_rate_base: [.0-9]+',
        #                'learning_rate_base: {}'.format("8e-2"), s)

        # Set warmup_learning_rate in learning_rate, sane default
        s = re.sub(
            "warmup_learning_rate: [.0-9]+", "warmup_learning_rate: {}".format(0.001), s
        )

        # Set warmup_steps in learning_rate, sane default
        s = re.sub("warmup_steps: [.0-9]+", "warmup_steps: {}".format(2500), s)

        # Set total_steps in learning_rate, num_steps
        s = re.sub("total_steps: [0-9]+", "total_steps: {}".format(num_train_steps), s)

        # Set number of classes num_classes.
        s = re.sub("num_classes: [0-9]+", "num_classes: {}".format(num_classes), s)

        # Setup the data augmentation preprocessor - not sure if this is a good one to use, commenting out for now and going with defaults.
        # s = re.sub('random_scale_crop_and_pad_to_square {\s+output_size: 896\s+scale_min: 0.1\s+scale_max: 2.0\s+}',
        #           'random_crop_image {\n\tmin_object_covered: 1.0\n\tmin_aspect_ratio: 0.75\n\tmax_aspect_ratio: 1.5\n\tmin_area: 0.25\n\tmax_area: 0.875\n\toverlap_thresh: 0.5\n\trandom_coef: 0.125\n}',s, flags=re.MULTILINE)

        # s = re.sub('ssd_random_crop {\s+}',
        #           'random_crop_image {\n\tmin_object_covered: 1.0\n\tmin_aspect_ratio: 0.75\n\tmax_aspect_ratio: 1.5\n\tmin_area: 0.10\n\tmax_area: 0.75\n\toverlap_thresh: 0.5\n\trandom_coef: 0.125\n}',s, flags=re.MULTILINE)

        # replacing the default data augmentation with something more comprehensive
        # the available options are listed here: https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto

        data_augmentation = (
            "data_augmentation_options {\n random_distort_color: { \n } \n}\n\n"
            "data_augmentation_options {\n random_horizontal_flip: { \n } \n}\n\n"
            "data_augmentation_options {\n random_vertical_flip: { \n } \n}\n\n"
            "data_augmentation_options {\n random_rotation90: { \n } \n}\n\n"
            "data_augmentation_options {\n random_jitter_boxes: { \n } \n}\n\n"
            "data_augmentation_options {\n random_crop_image {\n\tmin_object_covered: 1.0\n\tmin_aspect_ratio: 0.95\n\tmax_aspect_ratio: 1.05\n\tmin_area: 0.25\n\tmax_area: 0.875\n\toverlap_thresh: 0.9\n\trandom_coef: 0.5\n}\n}\n\n"
            "data_augmentation_options {\n random_jpeg_quality: {\n\trandom_coef: 0.5\n\tmin_jpeg_quality: 40\n\tmax_jpeg_quality: 90\n } \n}\n\n"
        )

        #https://github.com/tensorflow/models/issues/9379
        data_augmentation = (
            "data_augmentation_options {\n autoaugment_image: {\n } \n}\n\n"
        )

        s = re.sub(
            "data_augmentation_options {[\s\w]*{[\s\w\:\.]*}\s*}\s* data_augmentation_options {[\s\w]*{[\s\w\:\.]*}\s*}",
            data_augmentation,
            s,
            flags=re.MULTILINE,
        )

        # fine-tune checkpoint type
        s = re.sub(
            'fine_tune_checkpoint_type: "classification"',
            'fine_tune_checkpoint_type: "{}"'.format("detection"),
            s,
        )

        f.write(s)


def call_train_model(filepaths, num_train_steps, num_eval_steps):
    """Call model to initiate training."""
    pipeline_file = filepaths["pipeline_file"]
    model_dir = filepaths["model_dir"]

    command = """python /tf/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_file} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_train_steps} \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps={num_eval_steps}""".format(
        pipeline_file=pipeline_file,
        model_dir=model_dir,
        num_train_steps=num_train_steps,
        num_eval_steps=num_eval_steps,
    )

    subprocess.run(command.split(), check=True)
