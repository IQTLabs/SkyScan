import os
import time
import logging
import collections
import numpy as np

import fiftyone as fo

from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import label_map_util
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

Object = collections.namedtuple("Object", ["label", "score", "bbox"])


def _find_class_name(category_index, class_id):
    return category_index[class_id]["name"]


def _load_label_map(training_name):
    label_map_file = "/tf/dataset-export/" + training_name + "/label_map.pbtxt"
    label_map = label_map_util.load_labelmap(label_map_file)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=100
    )
    return label_map_util.create_category_index(categories)


def _tiles_location_gen(img_size, tile_size, overlap):
    """Generates location of tiles after splitting the given image according the tile_size and overlap.
    Args:
      img_size (int, int): size of original image as width x height.
      tile_size (int, int): size of the returned tiles as width x height.
      overlap (int): The number of pixels to overlap the tiles.
    Yields:
      A list of points representing the coordinates of the tile in xmin, ymin,
      xmax, ymax.
    """

    tile_width, tile_height = tile_size
    img_width, img_height = img_size
    h_stride = tile_height - overlap
    w_stride = tile_width - overlap
    for h in range(0, img_height, h_stride):
        for w in range(0, img_width, w_stride):
            xmin = w
            ymin = h
            xmax = min(img_width, w + tile_width)
            ymax = min(img_height, h + tile_height)
            yield [xmin, ymin, xmax, ymax]


def _non_max_suppression(objects, threshold):
    """Returns a list of indexes of objects passing the NMS.
    Args:
      objects: result candidates.
      threshold: the threshold of overlapping IoU to merge the boxes.
    Returns:
      A list of indexes containings the objects that pass the NMS.
    """
    if len(objects) == 1:
        return [0]

    boxes = np.array([o.bbox for o in objects])
    xmins = boxes[:, 0]
    ymins = boxes[:, 1]
    xmaxs = boxes[:, 2]
    ymaxs = boxes[:, 3]

    areas = (xmaxs - xmins) * (ymaxs - ymins)
    scores = [o.score for o in objects]
    idxs = np.argsort(scores)

    selected_idxs = []
    while idxs.size != 0:

        selected_idx = idxs[-1]
        selected_idxs.append(selected_idx)

        overlapped_xmins = np.maximum(xmins[selected_idx], xmins[idxs[:-1]])
        overlapped_ymins = np.maximum(ymins[selected_idx], ymins[idxs[:-1]])
        overlapped_xmaxs = np.minimum(xmaxs[selected_idx], xmaxs[idxs[:-1]])
        overlapped_ymaxs = np.minimum(ymaxs[selected_idx], ymaxs[idxs[:-1]])

        w = np.maximum(0, overlapped_xmaxs - overlapped_xmins)
        h = np.maximum(0, overlapped_ymaxs - overlapped_ymins)

        intersections = w * h
        unions = areas[idxs[:-1]] + areas[selected_idx] - intersections
        ious = intersections / unions

        idxs = np.delete(
            idxs, np.concatenate(([len(idxs) - 1], np.where(ious > threshold)[0]))
        )

    return selected_idxs


def _reposition_bounding_box(bbox, tile_location):
    """Relocates bbox to the relative location to the original image.
    Args:
      bbox (int, int, int, int): bounding box relative to tile_location as xmin,
        ymin, xmax, ymax.
      tile_location (int, int, int, int): tile_location in the original image as
        xmin, ymin, xmax, ymax.
    Returns:
      A list of points representing the location of the bounding box relative to
      the original image as xmin, ymin, xmax, ymax.
    """
    bbox[0] = bbox[0] + tile_location[0]
    bbox[1] = bbox[1] + tile_location[1]
    bbox[2] = bbox[2] + tile_location[0]
    bbox[3] = bbox[3] + tile_location[1]
    return bbox


def _get_resize(input_size, img_size):
    """Copies a resized and properly zero-padded image to a model's input tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` to update.
      size (tuple): The original image size as (width, height) tuple.
      resize: A function that takes a (width, height) tuple, and returns an
        image resized to those dimensions.
    Returns:
      The resized tensor with zero-padding as tuple
      (resized_tensor, resize_ratio).
    """
    width, height = input_size
    w, h = img_size
    scale = min(width / w, height / h)
    print(scale)
    w, h = int(w * scale), int(h * scale)
    return w, h
    tensor = input_tensor(interpreter)
    tensor.fill(0)  # padding
    _, _, channel = tensor.shape
    result = resize((w, h))
    tensor[:h, :w] = np.reshape(result, (h, w, channel))
    return result, (scale, scale)


def run_detection_model(dataset_name, training_name, prediction_field):
    model_path = (
        "/tf/model-export/" + training_name + "/image_tensor_saved_model/saved_model"
    )
    min_score = 0.8  # This is the minimum score for adding a prediction. This helps keep out bad predictions but it may need to be adjusted if your model is not that good yet.

    logging.info("Loading model...")
    start_time = time.time()
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(model_path)
    infer = detect_fn.signatures["serving_default"]
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Loading model took: " + str(elapsed_time) + "s")

    category_index = _load_label_map(training_name)

    dataset = fo.load_dataset(dataset_name)

    for sample in dataset.select_fields("filepath"):

        start_time = time.time()
        img = load_img(sample.filepath)
        img_array = img_to_array(img)
        input_tensor = np.expand_dims(img_array, 0)
        detections = detect_fn(input_tensor)
        exportDetections = []

        for i, detectScore in enumerate(detections["detection_scores"][0]):
            if detectScore > min_score:
                print(
                    "\t- {}: {}".format(
                        findClassName(int(detections["detection_classes"][0][i])),
                        detections["detection_scores"][0][i],
                    )
                )

                label = _find_class_name(int(detections["detection_classes"][0][i]))
                confidence = detections["detection_scores"][0][i]
                # TF Obj Detect bounding boxes are: [ymin, xmin, ymax, xmax]

                # For Voxel 51 - Bounding box coordinates should be relative values
                # in [0, 1] in the following format:
                # [top-left-x, top-left-y, width, height]
                x1 = detections["detection_boxes"][0][i][1]
                y1 = detections["detection_boxes"][0][i][0]
                x2 = detections["detection_boxes"][0][i][3]
                y2 = detections["detection_boxes"][0][i][2]
                w = x2 - x1
                h = y2 - y1
                bbox = [x1, y1, w, h]

                exportDetections.append(
                    fo.Detection(label=label, bounding_box=bbox, confidence=confidence)
                )

        # Store detections in a field name of your choice
        sample[prediction_field] = fo.Detections(detections=exportDetections)
        sample.save()
        end_time = time.time()

        print("Processing {} took: {}s".format(sample.filepath, end_time - start_time))


def run_detection_model_tiled(
    dataset_name,
    training_name,
    prediction_field,
    tile_string,
    tile_overlap,
    iou_threshold,
):

    model_path = (
        "/tf/model-export/" + training_name + "/image_tensor_saved_model/saved_model"
    )

    min_score = 0.8  # This is the minimum score for adding a prediction. This helps keep out bad predictions but it may need to be adjusted if your model is not that good yet.
    input_tensor_size = 512

    logging.info("Loading model...")
    start_time = time.time()
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(model_path)
    infer = detect_fn.signatures["serving_default"]
    print(infer.structured_outputs)
    print(infer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Loading model took: " + str(elapsed_time) + "s")

    category_index = _load_label_map(training_name)

    dataset = fo.load_dataset(dataset_name)

    # Go through all of the samples in the dataset
    for sample in dataset.select_fields("filepath"):

        start_time = time.time()
        img = load_img(
            sample.filepath,
        )
        img_size = img.size
        img_width, img_height = img_size
        objects_by_label = dict()
        exportDetections = []
        tile_sizes = []

        for tile_size in tile_string.split(","):
            tile_size = tile_size.split("x")
            tile_sizes.append([int(tile_size[0]), int(tile_size[1])])

        # Collect all of the detections for each tile size:
        for tile_size in tile_sizes:
            tile_width, tile_height = tile_size

            # For tiles that are smaller that the image size, calculated all of the different
            # Sub images that are needed
            for tile_location in _tiles_location_gen(img_size, tile_size, tile_overlap):

                tile = img.crop(tile_location)

                old_size = tile.size  # old_size[0] is in (width, height) format

                ratio = float(input_tensor_size) / max(old_size)
                if ratio > 1:
                    continue
                new_size = tuple([int(x * ratio) for x in old_size])

                im = tile.resize(new_size, Image.ANTIALIAS)
                # create a new image and paste the resized on it

                new_im = Image.new("RGB", (input_tensor_size, input_tensor_size))
                new_im.paste(
                    im, (0, 0)
                )  # ((input_tensor_size-new_size[0])//2, (input_tensor_size-new_size[1])//2))

                img_array = img_to_array(new_im, dtype="uint8")
                img_batch = np.array([img_array])

                detections = detect_fn(img_batch)
                for i, detectScore in enumerate(detections["detection_scores"][0]):
                    if detectScore > min_score:

                        x1 = (
                            detections["detection_boxes"][0][i][1].numpy()
                            * input_tensor_size
                        )  # tile_width
                        y1 = (
                            detections["detection_boxes"][0][i][0].numpy()
                            * input_tensor_size
                        )  # tile_height
                        x2 = (
                            detections["detection_boxes"][0][i][3].numpy()
                            * input_tensor_size
                        )  # tile_width
                        y2 = (
                            detections["detection_boxes"][0][i][2].numpy()
                            * input_tensor_size
                        )  # tile_height
                        bbox = [x1, y1, x2, y2]

                        scaled_bbox = []
                        for number in bbox:
                            scaled_bbox.append(number / ratio)
                        repositioned_bbox = _reposition_bounding_box(
                            scaled_bbox, tile_location
                        )
                        confidence = detections["detection_scores"][0][i]
                        label = _find_class_name(
                            category_index, int(detections["detection_classes"][0][i])
                        )
                        objects_by_label.setdefault(label, []).append(
                            Object(label, confidence, repositioned_bbox)
                        )

        for label, objects in objects_by_label.items():
            idxs = _non_max_suppression(objects, iou_threshold)
            for idx in idxs:
                x1 = objects[idx].bbox[0] / img_width
                y1 = objects[idx].bbox[1] / img_height
                x2 = objects[idx].bbox[2] / img_width
                y2 = objects[idx].bbox[3] / img_height

                w = x2 - x1
                h = y2 - y1
                bbox = [x1, y1, w, h]
                exportDetections.append(
                    fo.Detection(
                        label=objects[idx].label,
                        bounding_box=bbox,
                        confidence=objects[idx].score,
                    )
                )

        # Store detections in a field name of your choice
        sample[prediction_field] = fo.Detections(detections=exportDetections)
        sample.save()
        end_time = time.time()
        print("Processing {} took: {}s".format(sample.filepath, end_time - start_time))
