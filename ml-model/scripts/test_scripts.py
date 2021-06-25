"""test all scripts."""

import os

import fiftyone as fo

from customvox51 import (
    add_faa_data_to_voxel51_dataset,
    add_sample_images_to_voxel51_dataset,
    build_image_list,
    create_voxel51_dataset,
    normalize_single_model_value,
)
from detection import load_base_models_json, set_filenames

# from labelbox_utils import merge_labelbox_dataset_with_voxel51
from main import read_config

# pylint: disable=C0103, W0107

# delete dataset first to create repeatable test environment
try:
    fo.load_dataset("test").delete()
except ValueError:
    pass


def test_build_image_list():
    """Test build_image_list()."""
    output = build_image_list("test")
    assert output[0]["bearing"] == "194"
    assert output[0]["distance"] == "11882"
    assert output[0]["elevation"] == "50"
    assert output[0]["external_id"] == "ac760d_194_50_11882_2021-05-13-14-13-42"
    assert output[0]["icao24"] == "ac760d"


def test_create_voxel51_dataset():
    """Test create_voxel51_dataset()."""
    test_dataset = create_voxel51_dataset("test")
    assert isinstance(test_dataset, fo.core.dataset.Dataset)


def test_add_sample_images_to_voxel51_dataset():
    """Test add_sample_images_to_voxel51_dataset()."""
    test_image_list = build_image_list("test")
    test_dataset = create_voxel51_dataset("test")
    modified_dataset = add_sample_images_to_voxel51_dataset(
        test_image_list, test_dataset
    )
    assert isinstance(modified_dataset, fo.core.dataset.Dataset)


def test_add_faa_data_to_voxel51_dataset():
    """Test add_FAA_data_to_voxel51_dataset()."""
    test_image_list = build_image_list("test")
    test_dataset = create_voxel51_dataset("test")
    add_sample_images_to_voxel51_dataset(test_image_list, test_dataset)
    dataset_with_faa_data = add_faa_data_to_voxel51_dataset(
        "test", "../data/faa_master.txt", "../data/faa_aircraft_reference.txt"
    )
    assert isinstance(dataset_with_faa_data, fo.core.dataset.Dataset)
    assert dataset_with_faa_data.persistent
    assert dataset_with_faa_data.media_type == "image"


def test_read_config():
    """Test read_config()."""
    config = read_config(os.path.join("test", "test_config.ini"))
    assert config["file_names"]["dataset_name"] == "hello_world"
    assert config["file_locations"]["image_directory"] == "foo"
    assert config["file_locations"]["faa_aircraft_db"] == "bar"


def test_normalize_single_model_value():
    """Test normalize_single_model_value"""
    assert normalize_single_model_value("A320 232") == "A320"
    assert normalize_single_model_value("737-73V") == "737-700"


def test_normalize_model_values():
    """Test normalize_model_values()"""
    pass


def test_merge_labelbox_dataset_with_voxel51():
    """Test merge_labelbox_dataset_with_voxel51()"""
    # Writing a test for this function is difficult. Do if time allows.
    # test_dataset = create_voxel51_dataset("test")
    # merge_labelbox_dataset_with_voxel51(test_dataset, "test/labelbox_export_test.json")
    pass


def test_load_base_models_json():
    """Test load_base_models_json()"""
    test_models = load_base_models_json()
    assert test_models["ssd_mobilenet_v2"]["batch_size"] == 24
    assert (
        test_models["efficientdet-d0"]["pretrained_checkpoint"]
        == "efficientdet_d0_coco17_tpu-32.tar.gz"
    )


def test_set_filenames():
    """Test set_filenames()"""
    test_base_models = load_base_models_json()
    TEST_TRAINING_NAME = "luke_burnt"
    TEST_CHOSEN_MODEL = "ssd_mobilenet_v2"
    test_filepaths = set_filenames(
        test_base_models, TEST_TRAINING_NAME, TEST_CHOSEN_MODEL
    )
    assert test_filepaths["val_export_dir"] == "/tf/dataset-export/luke_burnt/val/"
    assert (
        test_filepaths["pretrained_checkpoint"]
        == "ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
    )
    assert (
        test_filepaths["pipeline_file"]
        == "/tf/models/research/deploy/ssd_mobilenet_v2_320x320_coco17_tpu-8.config"
    )
