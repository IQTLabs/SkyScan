# Plane Tracker

This container takes in a RTSP video feed, use the Google Coral Accelerator to detect a plane and then generates an MQTT message steering the camera to the center of the detected bounding box.

Things to do:
- describe how to config the Axis camera feed. You need to an MJPEG feed. An MP4 based feed has too much latency.
- Need to make this into a container






# Edge TPU Object Tracker Example

This repo contains a collection of examples that use camera streams
together with the [TensorFlow Lite API](https://tensorflow.org/lite) with a
Coral device such as the
[USB Accelerator](https://coral.withgoogle.com/products/accelerator) or
[Dev Board](https://coral.withgoogle.com/products/dev-board) and provides an Object tracker for use with the detected objects.


## Installation

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)).

2.  Clone this Git repo onto your computer:

    ```
    mkdir google-coral && cd google-coral

    git clone https://github.com/google-coral/example-object-tracker.git

    cd example-object-tracker/
    ```

3.  Download the models:

    ```
    sh download_models.sh
    ```

    These models will be downloaded to a new folder
    ```models```.


Further requirements may be needed by the different camera libraries, check the
README file for the respective subfolder.

## Contents

  * __gstreamer__: Python examples using gstreamer to obtain camera streem. These
    examples work on Linux using a webcam, Raspberry Pi with
    the Raspicam and on the Coral DevBoard using the Coral camera. For the
    former two you will also need a Coral USB Accelerator to run the models.

    This demo provides the support of an Object tracker. After following the setup 
    instructions in README file for the subfolder ```gstreamer```, you can run the tracker demo:

    ```
    cd gstreamer
    python3 detect.py --tracker sort
    ```

## Models

For the demos in this repository you can change the model and the labels
file by using the flags flags ```--model``` and
```--labels```. Be sure to use the models labeled _edgetpu, as those are
compiled for the accelerator -  otherwise the model will run on the CPU and
be much slower.


For detection you need to select one of the SSD detection models
and its corresponding labels file:

```
mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite, coco_labels.txt
```


