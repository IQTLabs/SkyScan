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
import errno
import paho.mqtt.client as mqtt 
import pantilthat

args = None
pan = 0
tilt = 0

#############################################
##         MQTT Callback Function          ##
#############################################
def on_message(client, userdata, message):
    command = str(message.payload.decode("utf-8"))
    update = json.loads(command)
    logging.info(update)
    logging.info("Bearing: $d Azimuth: $f" % update["bearing"] % update["azimuth"])
    if (update["bearing"] < 90):
        logging.info("Setting Pan to: $d" % update["bearing"])
        pantilthat.pan(update["bearing"])
    if (update["bearing"] > 270):
        logging.info("Setting Pan to: $d" % update["bearing"])
        pantilthat.pan(update["bearing"]-360)
    if (update["azimuth"] < 90):
        logging.info("Setting Tilt to: $f" % update["azimuth"])
        pantilthat.tilt(update["azimuth"])


def main():
    global args
    global logging
    global pan
    global tilt
    parser = argparse.ArgumentParser(description='An MQTT based camera controller')

    parser.add_argument('-b', '--bearing', help="What bearing is the font of the PI pointed at (0-360)", default=0)
    parser.add_argument('-m', '--mqtt-host', help="MQTT broker hostname", default='127.0.0.1')
    parser.add_argument('-t', '--mqtt-topic', help="MQTT topic to subscribe to", default="SkyScan")
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
    pantilthat.pan(pan)
    pantilthat.tilt(tilt)
        # Sleep for a bit so we're not hammering the HAT with updates
    time.sleep(0.005)
    print("connecting to MQTT broker at "+ args.mqtt_host+", channel '"+args.mqtt_topic+"'")
    client = mqtt.Client("pan-tilt-pi-camera") #create new instance

    client.on_message=on_message #attach function to callback

    client.connect(args.mqtt_host) #connect to broker
    client.loop_start() #start the loop
    client.subscribe(args.mqtt_topic+"/#")
    #############################################
    ##                Main Loop                ##
    #############################################
    while True:
        time.sleep(0.1)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e, exc_info=True)
