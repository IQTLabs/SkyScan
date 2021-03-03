# OpenCV camera examples with Coral

This folder contains example code using [OpenCV](https://github.com/opencv/opencv) to obtain
camera images and perform object detection on the Edge TPU.

This code works on Linux/macOS/Windows using a webcam, Raspberry Pi with the Pi Camera, and on the Coral Dev
Board using the Coral Camera or a webcam. For all settings other than the Coral Dev Board, you also need a Coral
USB/PCIe/M.2 Accelerator.


## Set up your device

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)). You can check which version is installed
    using the ```pip3 show tflite_runtime``` command.

1.5 Install PyCoral: https://coral.ai/software/#pycoral-api


2.  Clone this Git repo onto your computer or Dev Board:

    ```
    mkdir google-coral && cd google-coral

    git clone https://github.com/google-coral/examples-camera --depth 1
    ```

3.  Download the models:

    ```
    cd examples-camera

    sh download_models.sh
    ```

4.  Install the OpenCV libraries:

    ```
    cd opencv

    bash install_requirements.sh
    ```


## Run the detection model with Sort tracker
```
python3 detect.py --tracker sort
```

## Run the detection demo without any tracker (SSD models)

```
python3 detect.py
```

## Arguments

*All of the arguments are optional and provide increasing control over the configuration*

 - **model** path to the model you want to use, defaults to COCO
 - **labels** labels for the model you are using, default to COCO labels
 - **top_k**  number of categories with highest score to display, defaults to 3
 - **threshold** classifier score threshold
 - **videosrc** what video source you want to use. Choices are `net` or `dev`. Default is `dev`:
    - **dev** a directly connected (dev) camera, can be Coral cam or USB cam or Networked 
    - **net** network video source, using RTSP. The --netsrc argument must be specified. 
	- **file** a video file can be used as a source
 - **camera_idx**  Index of which video source to use. I am not sure how OpenCV enumerates them. Defaults to 0.
 - **filesrc** the path to the video file. In the Docker container should be at /app/videos
 - **netsrc** If the `videosrc` is `net` then specify the URL. Example: `rtsp://192.168.1.43/mpeg4/media.amp`
 - **tracker** Name of the Object Tracker To be used. Choices are `None` or `sort`.
 
You can change the model and the labels file using ```--model``` and ```--labels```.

By default, this uses the ```mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite``` model.

You can change the model and the labels file using flags ```--model``` and ```--labels```.
