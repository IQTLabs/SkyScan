# SkyScan
Automatically photograph planes that fly by!
-----
Follow allowing as we build, [here](https://iqtlabs.github.io/SkyScan/)

![Airbus A321](media/a321.jpg)
*Airbus A321 at 32,000 feet*

## Overview
To enable better tracking, most planes broadcast a signal known as [Automatic Dependent Surveillance–Broadcast](https://en.wikipedia.org/wiki/Automatic_Dependent_Surveillance–Broadcast) or ADS-B. This signal is at 1090MHz and can be easily received using a low cost Software Defined Radio (SDR), like the [RTL-SDR](https://learn.adafruit.com/getting-started-with-rtl-sdr-and-sdr-sharp) which repurposes a digital TV chip.

From the ADS-B transmissions, you can get a plane's location and altitude. If you know where a plane is and where you are, you can do some math and point a camera at the plane and take a picture. If you have a Pan/Tilt camera lying around, you can have it automatically track a plane as it flies by and snap photos.

## Hardware
The project is built around the RaspberryPi 4, an RTL-SDR, and an Axis PTZ security camera. It should easily be extended to work with other SDRs or cameras.

Axis has a great API for their network cameras, and it should work with any of there PTZ cameras. The m5525 is nice because it supports continuous 360 degree rotation. You can literally have it spin around in circles, giving you complete coverage. The code has been tested with the 10x zoom [Axis m5525](https://www.axis.com/en-us/products/axis-m5525-e) and the 30x zoom [Axis p5655](https://www.axis.com/en-us/products/axis-p5655-e) cameras.

We are using the [Nooelec NESDR SMArt v4 SDR](https://www.nooelec.com/store/sdr/nesdr-smart-sdr.html) This is nice and stable RTL-SDR. It is compact and doesn't block all the other ports on a Pi. Since you are just trying to capture local planes, you can get away with using any antenna you have lying around.

### Field System Configurations
Two configurations were developed for this project, one with AC and one with DC power input.

![configurations](media/hardware1-small.JPG)

See the [Hardware README](hardware/README.md) for additional details including a BOM.

## Software Architecture

The different components for this project have been made into Docker containers. This modularity makes it easier to add in new data sources or cameras down the road. We have found containers to work really well on the Pi and the help enforce that you have properly documented all of the software requirements.

````
+-------------+      +-------------+           +---------------+            +--------------+
|             |      |             |           |               |            |              |
|             |      |             |           |               |            |  Axis+PTZ    |
| Pi+Aware    +----->+  ADSB+MQTT  +---------->+ Tracker       +----------->+              |
|             | TCP  |             |  MQTT     |               |  MQTT      |              |
|             |      |             |  all      |               |  only      |              |
+-------------+      +-------------+  planes   +---------------+  tracking  +-------+------+
                                                                  plane             |
                                                                                    | HTTP API
                                                                                    | Pan/Tilt
                                                                                    v
                                +--------------------+                     +--------+---------+
                                |                    |                     |                  |
                                |                    |                     |                  |
                                |   MQTT Broker      |                     |    Axis m5525    |
                                |                    |                     |    Camera        |
                                |                    |                     |                  |
                                |                    |                     |                  |
                                +--------------------+                     +------------------+

````

Here is a brief overview of each component. There are additional details in the component's subdirectory

- [mikenye/piaware](https://github.com/mikenye/docker-piaware) - This is a dockerized version of FlightAware's [PiAware](https://flightaware.com/adsb/piaware/) program. PiAware is actually just a wrapper around [dump1090](https://flightaware.com/adsb/piaware/). Dump1090 is a small program that can use an RTL-SDR to receive an ADS-B transmission. The program uses these transmission to track where nearby planes are and display then on a webpage. It also output all of the messages it receives on a TCP port, for other programs to use. We use this connection to get the plane information. PiAware adds the ability to send the information to FlightAware. You could probably just switch this to only use Dump1090.

- [ADSB-MQTT](adsb-mqtt) Is a small python program that reads in information collected by Dump1090 over a TCP port and publishes all the messages it receives onto the MQTT bus. 

- [tracker](tracker) Receives all of the plane's location, determines how far away from the camera each one is and then finds the closest plane. The location and relative position of the closest plane is periodically published as an MQTT messages. Tracker needs to know the location and altitude of the camera in order to determine the planes relative position.

- [axis-ptz](axis-ptz) Receives updates on which plane to track over MQTT and then directs the PTZ camera towards the plane and takes a picture.

## Installation

### Install and Configure Raspberry Pi

Follow the instructions here: [configure-pi.md](./configure-pi.md)

### Configure PiAware

In order to start PiAware, you need to register with Flight Aware and request a Feed ID. There are directions on how to do that [here](https://github.com/mikenye/docker-piaware#new-to-piaware).

Copy the **env-example** file to **.env**. Edit the **.env** file to include the correct values.

### Configure Camera

#### Axis Security Camera

- Locate your camera on your network using the [Axis Discovery Tool](https://www.axis.com/support/downloads/axis-ip-utility) or with the following linux command `avahi-browse -a -r`
- Add IP and login info to `.env` file 
> AXIS_USERNAME= # The username for the Axis camera
> AXIS_PASSWORD= # The Password for the Axis camera
> AXIS_IP= # The IP address of the camera

#### (Optional) Enable Raspi-camera

If you are using the Pan Tilt hat, you will need to make sure the Pi Camera has been configured correctly:
In the base OS on the Pi make sure the Camera is enabled:
```bash
sudo raspi-config
```
- Interfacing Options
- Camera
- Enable

## System Operation

### Hardware Setup/Operation
1. Unpack and connect system components

<img src="media/ac-components-2.jpg" alt="components" title="components" width="400" />

2. Level tripod and adjustable head using bubble levels

<img src="media/ac-components-7.jpg" alt="components" title="components" width="400" />

3. Orient camera so it is pointed north

<img src="media/ac-components-4.jpg" alt="components" title="components" width="400" />

4. Connect the AC adapter to the back of the PoE switch

<img src="media/ac-components-5.jpg" alt="components" title="components" width="400" />

4. Power on the system (optional use of portable battery station for remote operation)

<img src="media/ac-components-6.jpg" alt="components" title="components" width="400" />

### Software Operation

SSH to the RaspberryPi and launch the application using docker-compose: 
```bash
cd ~/Projects/SkyScan
docker-compose up
```

A web interface will be available on **port 8080**. As pictures of planes are captured they will be saved in folders in the **./capture** directory.

#### Testing with pytest

To run tests with pytest, run:

```bash
pytest
```


