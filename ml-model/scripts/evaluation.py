import logging
import fiftyone as fo
from fiftyone import ViewField as F
import os
import numpy as np
import matplotlib.pyplot as plt

def evaluate_detection_model(dataset_name, prediction_field, evaluation_key, ground_truth_field):

    dataset = fo.load_dataset(dataset_name)

    view = dataset.match_tags("multi_class_eval")

    # setting an empty detections field if there isn't one
    for sample in view:
        if sample[ground_truth_field] == None:
            sample[ground_truth_field] = fo.Detections(detections=[])
            sample.save()
        if sample[prediction_field] == None:
            sample[prediction_field] = fo.Detections(detections=[])
            sample.save()

    results = view.evaluate_detections(
        prediction_field, gt_field=ground_truth_field, eval_key=evaluation_key, compute_mAP=True
    )

    # Get the 10 most common classes in the dataset
    counts = view.count_values("{}.detections.label".format(ground_truth_field))
    classes = sorted(counts, key=counts.get, reverse=True)[:15]

    # Print a classification report for the top-10 classes
    results.print_report(classes=classes)

    # Print some statistics about the total TP/FP/FN counts
    logging.info("TP: %d" % dataset.sum(evaluation_key + "_tp"))
    logging.info("FP: %d" % dataset.sum(evaluation_key + "_fp"))
    logging.info("FN: %d" % dataset.sum(evaluation_key + "_fn"))

    # Create a view that has samples with the most false positives first, and
    # only includes false positive boxes in the `predictions` field
    eval_view = view.sort_by(evaluation_key + "_fp", reverse=True).filter_labels(
        prediction_field, F(evaluation_key) == "fp"
    )
    logging.info("mAP: {}".format(results.mAP()))


    plot = results.plot_pr_curves(classes=classes,backend="matplotlib")
    plot.savefig( "/tf/dataset-export/"+ evaluation_key +  '_pr_curves.png')

    plot = results.plot_confusion_matrix(classes=classes,backend="matplotlib")
    plot.savefig( "/tf/dataset-export/"+ evaluation_key + '_confusion_matrix.png')


