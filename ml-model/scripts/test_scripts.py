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
from detection import (
    create_detection_mapping,
    download_base_training_config,
    export_voxel51_dataset_to_tfrecords,
    load_base_models_json,
    save_mapping_to_file,
    set_filenames,
    _create_list_of_class_names,
)

# from labelbox_utils import merge_labelbox_dataset_with_voxel51
from main import read_config

# pylint: disable=C0103, W0107

# delete datasets first to create repeatable test environment
try:
    fo.load_dataset("test").delete()
except ValueError:
    pass
try:
    fo.load_dataset("test_detection_mapping").delete()
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
        test_image_list, test_dataset, "test-data"
    )
    assert isinstance(modified_dataset, fo.core.dataset.Dataset)
    assert modified_dataset.first()["tags"] == ["test-data"]


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


def test_export_voxel51_dataset_to_tfrecords():
    """Test export_voxel51_dataset_to_tfrecords()."""
    # needed for set up
    create_voxel51_dataset("test")
    test_base_models = load_base_models_json()
    TEST_TRAINING_NAME = "luke_burnt"
    TEST_CHOSEN_MODEL = "ssd_mobilenet_v2"
    test_filepaths = set_filenames(
        test_base_models, TEST_TRAINING_NAME, TEST_CHOSEN_MODEL
    )

    export_voxel51_dataset_to_tfrecords(
        "test", test_filepaths, label_field="detections"
    )


def test__create_list_of_class_names():
    """Test _create_list_of_class_names."""
    # credit to voxel51 crew for a helpful test suite from which this test borrows
    # https://github.com/voxel51/fiftyone/blob/a7c2b36a4f101330fa8edec35a9bdee841886f96/tests/unittests/view_tests.py#L59
    dataset = fo.Dataset()
    dataset.add_sample(
        fo.Sample(
            filepath="filepath1.jpg",
            tags=["test"],
            test_dets=fo.Detections(
                detections=[
                    fo.Detection(
                        label="friend",
                        confidence=0.9,
                        bounding_box=[0, 0, 0.5, 0.5],
                    ),
                    fo.Detection(
                        label="stopper",
                        confidence=0.1,
                        bounding_box=[0, 0, 0.5, 0.5],
                    ),
                    fo.Detection(
                        label="big bro",
                        confidence=0.6,
                        bounding_box=[0, 0, 0.1, 0.5],
                    ),
                ]
            ),
            another_field=51,
        )
    )
    test_list = _create_list_of_class_names(dataset, label_field="test_dets")
    assert set(test_list) == set(["friend", "stopper", "big bro"])


def test_create_detection_mapping():
    """Test create_detection_mapping()."""
    # credit to voxel51 crew for a helpful test suite from which this test borrows
    # https://github.com/voxel51/fiftyone/blob/a7c2b36a4f101330fa8edec35a9bdee841886f96/tests/unittests/view_tests.py#L59
    dataset = fo.Dataset(name="test_detection_mapping")
    dataset.add_sample(
        fo.Sample(
            filepath="filepath1.jpg",
            tags=["training"],
            test_dets=fo.Detections(
                detections=[
                    fo.Detection(
                        label="friend",
                        confidence=0.9,
                        bounding_box=[0, 0, 0.5, 0.5],
                    ),
                    fo.Detection(
                        label="stopper",
                        confidence=0.1,
                        bounding_box=[0, 0, 0.5, 0.5],
                    ),
                ]
            ),
            another_field=51,
        )
    )
    test_output = create_detection_mapping(
        "test_detection_mapping", label_field="test_dets"
    )
    assert isinstance(test_output, str)
    assert (
        test_output
        == 'item {\n  name: "friend"\n  id: 1\n}\nitem {\n  name: "stopper"\n  id: 2\n}\n'
    )


def test_save_mapping_to_file():
    """Test save_mapping_to_file()."""
    test_base_models = load_base_models_json()
    TEST_TRAINING_NAME = "luke_burnt"
    TEST_CHOSEN_MODEL = "ssd_mobilenet_v2"
    test_filepaths = set_filenames(
        test_base_models, TEST_TRAINING_NAME, TEST_CHOSEN_MODEL
    )
    test_mapping = str(
        'item {\n  name: "friend"\n  id: 1\n}\nitem {\n  name: "stopper"\n  id: 2\n}\n'
    )

    save_mapping_to_file(test_mapping, test_filepaths)

    with open(test_filepaths["label_map_file"], "r") as test_file:
        assert (
            test_file.read()
            == 'item {\n  name: "friend"\n  id: 1\n}\nitem {\n  name: "stopper"\n  id: 2\n}\n'
        )


def test_download_base_training_config():
    """Test download_base_training_config()."""
    # pylint: disable=line-too-long
    test_base_models = load_base_models_json()
    TEST_TRAINING_NAME = "luke_burnt"
    TEST_CHOSEN_MODEL = "ssd_mobilenet_v2"
    test_filepaths = set_filenames(
        test_base_models, TEST_TRAINING_NAME, TEST_CHOSEN_MODEL
    )
    download_base_training_config(test_filepaths)
    assert os.path.isfile(
        "/tf/models/research/deploy/" + test_filepaths["base_pipeline_file"]
    )
