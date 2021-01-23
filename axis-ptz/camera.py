#!/usr/bin/env python3



import argparse
import threading
import json
import sys
import os
import logging
import logging
import coloredlogs
import calendar
from datetime import datetime, timedelta
import signal
import random
import time
import re
import requests
from requests.auth import HTTPDigestAuth
import errno
import paho.mqtt.client as mqtt 
from json.decoder import JSONDecodeError
from sensecam_control import vapix_control,vapix_config


ID = str(random.randint(1,100001))
tiltCorrect = 15
args = None
camera = None
cameraConfig = None
cameraZoom = None
cameraMoveSpeed = None
cameraDelay = None
pan = 0
tilt = 0
actualPan = 0
actualTilt = 0
currentPlane=None

# https://stackoverflow.com/questions/45659723/calculate-the-difference-between-two-compass-headings-python

# The camera hat takes bearings between -90 and 90. 
# h1 is the target heading
# h2 is the heading the camera is pointed at
def getHeadingDiff(h1, h2):
    if h1 > 360 or h1 < 0 or h2 > 360 or h2 < 0:
        raise Exception("out of range")
    diff = h1 - h2
    absDiff = abs(diff)

    if absDiff == 180:
        return absDiff
    elif absDiff < 180:
        return diff
    elif h2 > h1:
        return 360 - absDiff
    else:
        return absDiff - 360

def setPan(bearing):
    global pan
    camera_bearing = args.bearing
    diff_heading = getHeadingDiff(bearing, camera_bearing)
    

    if pan != bearing: #abs(pan - diff_heading) > 2: #only update the pan if there has been a big change
        #logging.info("Heading Diff %d for Bearing %d & Camera Bearing: %d"% (diff_heading, bearing, camera_bearing))

        pan = bearing #diff_heading
        #logging.info("Setting Pan to: %d"%pan)
            
        return True
    return False

def setTilt(elevation):
    global tilt
    if elevation < 90:
        if tilt != elevation: #abs(tilt-elevation) > 2:
            tilt = elevation
            
            #logging.info("Setting Tilt to: %d"%elevation)

 # Copied from VaPix/Sensecam to customize the folder structure for saving pictures          
def get_jpeg_request():  # 5.2.4.1
    """
    The requests specified in the JPEG/MJPG section are supported by those video products
    that use JPEG and MJPG encoding.
    Args:
        resolution: Resolution of the returned image. Check the product’s Release notes.
        camera: Selects the source camera or the quad stream.
        square_pixel: Enable/disable square pixel correction. Applies only to video encoders.
        compression: Adjusts the compression level of the image.
        clock: Shows/hides the time stamp. (0 = hide, 1 = show)
        date: Shows/hides the date. (0 = hide, 1 = show)
        text: Shows/hides the text. (0 = hide, 1 = show)
        text_string: The text shown in the image, the string must be URL encoded.
        text_color: The color of the text shown in the image. (black, white)
        text_background_color: The color of the text background shown in the image.
        (black, white, transparent, semitransparent)
        rotation: Rotate the image clockwise.
        text_position: The position of the string shown in the image. (top, bottom)
        overlay_image: Enable/disable overlay image.(0 = disable, 1 = enable)
        overlay_position:The x and y coordinates defining the position of the overlay image.
        (<int>x<int>)
    Returns:
        Success ('image save' and save the image in the file folder) or Failure (Error and
        description).
    """
    payload = {
        'resolution': "1920x1080",
        'camera': 1,
    }
    url = 'http://' + args.axis_ip + '/axis-cgi/jpg/image.cgi'
    resp = requests.get(url, auth=HTTPDigestAuth(args.axis_username, args.axis_password),
                        params=payload)

    if resp.status_code == 200:
        captureDir = "capture/{}".format(currentPlane["type"])
        try:
            os.makedirs(captureDir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..
        filename = "{}/{}_{}.jpg".format(captureDir, currentPlane["icao24"],datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        
        with open(filename, 'wb') as var:
            var.write(resp.content)
        return str('Image saved')

    text = str(resp)
    text += str(resp.text)
    return text


def moveCamera():
    global actualPan
    global actualTilt
    global camera

    
    while True:
        lockedOn = False
        if actualTilt != tilt or actualPan != pan:
            logging.info("Moving camera to Tilt: %d  & Pan: %d"%(tilt, pan))
            actualTilt = tilt
            actualPan = pan
            lockedOn = True
            camera.absolute_move(pan, tilt, 9999, 50)
            time.sleep(0.3)
            get_jpeg_request()
                

        #if lockedOn == True:
        #    filename = "capture/{}_{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), currentPlane)
        #    camera.capture("{}.jpeg".format(filename))

        # Sleep for a bit so we're not hammering the camera withupdates
        time.sleep(0.005)

#############################################
##         MQTT Callback Function          ##
#############################################
def on_message(client, userdata, message):
    global currentPlane
    command = str(message.payload.decode("utf-8"))
    #rint(command)
    try:
        update = json.loads(command)
        #payload = json.loads(messsage.payload) # you can use json.loads to convert string to json
    except JSONDecodeError as e:
    # do whatever you want
        print(e)
    except TypeError as e:
    # do whatever you want in this case
        print(e)
    except ValueError as e:
        print(e)
    except:
        print("Caught it!")
    
    logging.info("{}\tBearing: {} \tElevation: {}".format(update["icao24"],update["bearing"],update["elevation"]))
    bearingGood = setPan(update["bearing"])
    setTilt(update["elevation"])
    currentPlane = update

def main():
    global args
    global logging
    global pan
    global tilt
    global camera
    global cameraDelay
    global cameraMoveSpeed
    global cameraZoom
    global cameraConfig

    parser = argparse.ArgumentParser(description='An MQTT based camera controller')

    parser.add_argument('-b', '--bearing', help="What bearing is the font of the PI pointed at (0-360)", default=0)
    parser.add_argument('-m', '--mqtt-host', help="MQTT broker hostname", default='127.0.0.1')
    parser.add_argument('-t', '--mqtt-topic', help="MQTT topic to subscribe to", default="SkyScan")
    parser.add_argument('-u', '--axis-username', help="Username for the Axis camera", required=True)
    parser.add_argument('-p', '--axis-password', help="Password for the Axis camera", required=True)
    parser.add_argument('-a', '--axis-ip', help="IP address for the Axis camera", required=True)
    parser.add_argument('-s', '--camera-move-speed', type=int, help="The speed at which the Axis will move for Pan/Tilt (0-100)", default=50)
    parser.add_argument('-d', '--camera-delay', type=float, help="How many seconds after issuing a Pan/Tilt command should a picture be taken", default=0.5)
    parser.add_argument('-z', '--camera-zoom', type=int, help="The zoom setting for the camera (0-9999)", default=9999)
    parser.add_argument('-v', '--verbose',  action="store_true", help="Verbose output")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    
    styles = {'critical': {'bold': True, 'color': 'red'}, 'debug': {'color': 'green'}, 'error': {'color': 'red'}, 'info': {'color': 'white'}, 'notice': {'color': 'magenta'}, 'spam': {'color': 'green', 'faint': True}, 'success': {'bold': True, 'color': 'green'}, 'verbose': {'color': 'blue'}, 'warning': {'color': 'yellow'}}
    level = logging.DEBUG if '-v' in sys.argv or '--verbose' in sys.argv else logging.INFO
    if 1:
        coloredlogs.install(level=level, fmt='%(asctime)s.%(msecs)03d \033[0;90m%(levelname)-8s '
                            ''
                            '\033[0;36m%(filename)-18s%(lineno)3d\033[00m '
                            '%(message)s',
                            level_styles = styles)
    else:
        # Show process name
        coloredlogs.install(level=level, fmt='%(asctime)s.%(msecs)03d \033[0;90m%(levelname)-8s '
                                '\033[0;90m[\033[00m \033[0;35m%(processName)-15s\033[00m\033[0;90m]\033[00m '
                                '\033[0;36m%(filename)s:%(lineno)d\033[00m '
                                '%(message)s')

    logging.info("---[ Starting %s ]---------------------------------------------" % sys.argv[0])
    camera = vapix_control.CameraControl(args.axis_ip, args.axis_username, args.axis_password)
    cameraDelay = args.camera_delay
    cameraMoveSpeed = args.camera_move_speed
    cameraZoom = args.camera_zoom
    cameraConfig = vapix_config.CameraConfiguration(args.axis_ip, args.axis_username, args.axis_password)

    threading.Thread(target = moveCamera, daemon = True).start()
        # Sleep for a bit so we're not hammering the HAT with updates
    time.sleep(0.005)
    print("connecting to MQTT broker at "+ args.mqtt_host+", channel '"+args.mqtt_topic+"'")
    client = mqtt.Client("skyscan-axis-ptz-camera-" + ID) #create new instance

    client.on_message=on_message #attach function to callback

    client.connect(args.mqtt_host) #connect to broker
    client.loop_start() #start the loop
    client.subscribe(args.mqtt_topic+"/#")
    client.publish("skyscan/registration", "skyscan-axis-ptz-camera-"+ID+" Registration", 0, False)

    #############################################
    ##                Main Loop                ##
    #############################################
    timeHeartbeat = 0
    while True:
        if timeHeartbeat < time.mktime(time.gmtime()):
            timeHeartbeat = time.mktime(time.gmtime()) + 10
            client.publish("skyscan/heartbeat", "skyscan-axis-ptz-camera-"+ID+" Heartbeat", 0, False)
        time.sleep(0.1)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e, exc_info=True)
