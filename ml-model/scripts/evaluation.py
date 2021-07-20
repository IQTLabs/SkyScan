import logging
import fiftyone as fo
from fiftyone import ViewField as F
import os
import numpy as np

def evaluate_detection_model(dataset_name, prediction_field, evaluation_key):

    dataset = fo.load_dataset(dataset_name) 

    view = dataset.match_tags(evaluation_name)

    # setting an empty detections field if there isn't one
    for sample in view:
        if sample["detections"] == None:
            sample["detections"] = fo.Detections(detections=[])
            sample.save()
    
    results = view.evaluate_detections( prediction_field, gt_field="detections", eval_key=evaluation_key)

    # Get the 10 most common classes in the dataset
    counts = view.count_values("detections.detections.label")
    classes = sorted(counts, key=counts.get, reverse=True)[:10]

    # Print a classification report for the top-10 classes
    results.print_report(classes=classes)

    # Print some statistics about the total TP/FP/FN counts
    logging.info("TP: %d" % dataset.sum(evaluation_key + "_tp"))
    logging.info("FP: %d" % dataset.sum(evaluation_key + "_fp"))
    logging.info("FN: %d" % dataset.sum(evaluation_key + "_fn"))

    # Create a view that has samples with the most false positives first, and
    # only includes false positive boxes in the `predictions` field
    eval_view = (view
        .sort_by(evaluation_key + "_fp", reverse=True)
        .filter_labels(prediction_field, F(evaluation_key) == "fp")
    )
    logging.info("mAP: {}".format(results.mAP()))