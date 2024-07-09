# Training the PyTorch Localizer and Detector

## Content

The `model.py` files contains a Python class called `Dataset`, which has methods for manipulating datasets of SkyScan images and training models on them.  There are also some example functions that use the class, including `revised_sequence()`, which runs an entire data analysis pipeline.  The steps in that pipeline including ingesting the imagery, training a localizer (which finds bounding boxes), using the localizer to find bounding boxes on additional images, training a detector (which finds bounding boxes and aircraft class), and testing the performance of localizer and detector with testing data.

The pipeline will need to be adapted to meet your specific needs.  Each time a model is trained or used, YOLOv7 outputs the results to a folder.  To retrain a model or rerun a test, the corresponding folder name should be changed in the code to avoid a collision.  On the other hand, if you've already trained a model or run a test and are rerunning the code, you can instead comment out the corresponding `.train()` and `.test()` calls to save time.  In general, lines ending with `##` in `revised_sequence()` are time-intensive steps that only need to be run once, and can be commented out to save time if rerunning the code.

## Setup

Install the [YOLOv7 repo](https://github.com/WongKinYiu/yolov7), following the instructions there.  Then, copy the following files into the root folder of your YOLOv7 installation:
* `model.py`
* `test2.py`
* `localizer_rev2.yaml`
* `detector_rev2.yaml`
* `taxon.json`

Copy the following files into the `utils` folder:
* `metrics2.py`
* `datasets2.py`

Then, edit the `model.py` file's `revised_sequence()` method and update the file paths to point to the appropriate folders for your system.  File paths in `localizer_rev2.yaml` and `detector_rev2.yaml` will also need to be updated.

Finally, from the YOLOv7 root folder, run:
`./model.py`

## License

The contents of this folder are derived from https://github.com/WongKinYiu/yolov7 and are being released under the GNU General Public License v3.0. (The remainder of this repo is released under the Apache License 2.0.)
