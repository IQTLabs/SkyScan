#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Johan Kanflo (github.com/kanflo)
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#
# Subscribes to the topic 'adsb/proximity/json' and publishes a color to the
# topic 'ghost/led' according to operator/distance of an aircraft.
#

import imagecolor
import os
import paho.mqtt.client as mosquitto
import Queue
import time
from threading import *
import remotelogger
import logging, sys
import socket
import calendar, datetime
import json
import traceback
import math
import signal
import random
import argparse
import bing

gQuitting = False
gCurrentColor = ()

gConnectCount = 0

log = logging.getLogger(__name__)


def mqttOnConnect(mosq, obj, rc):
    log.info("MQTT connected")
    mosq.subscribe("adsb/proximity/json", 0)
    log.debug("MQTT Connect: %s" % (str(rc)))

def mqttOnDisconnect(mosq, obj, rc):
    if 1:
        log.info("MQTT disconnected")
    else:
        global gQuitting
        global gConnectCount
        log.info("MQTT Disconnect: %s" % (str(rc)))
        gConnectCount += 1
        if gConnectCount == 10:
            log.info("Giving up!")
            gQuitting = True
            sys.exit()
        if not gQuitting:
            while not mqttConnect():
                time.sleep(10)
                log.info("Attempting MQTT reconnect")
        log.info("MQTT connected")

def mqttOnMessage(mosq, obj, msg):
    global args
    global gCurrentColor
    try:
        data = json.loads(msg.payload)
    except Exception as e:
        log.error("JSON load failed for '%s' : %s" % (msg.payload, e))
        print(traceback.format_exc())
        return

    if data["operator"] and data["distance"]:
        airline = data["operator"]
        distance = data["distance"]
        lost = False
        if "lost" in data:
            lost = data["lost"]

        if airline == "SAS":
            airline = "SAS Airlines"

        if lost:
            log.debug("Lost sight of aircraft")
            color = (0, 0, 0)
        else:
            try:
                color = imagecolor.getColor(airline)
            except Exception as e:
                log.error("getColor failed  %s" % (e))
                print(traceback.format_exc())
                return
        if distance > args.max_distance or not color:
            color = (0, 0, 0)
        else:
            color_0 = int(color[0] * (1 - (distance / args.max_distance)))
            color_1 = int(color[1] * (1 - (distance / args.max_distance)))
            color_2 = int(color[2] * (1 - (distance / args.max_distance)))
            color = (color_0, color_1, color_2)
        if color != gCurrentColor:
            log.debug("New color is %02x%02x%02x" % (color[0], color[1], color[2]))
            # This is a bit lazy, I know...
            cmd = "mosquitto_pub -h %s -t %s -m \"#%02x%02x%02x\"" % (args.mqtt_host, args.mqtt_topic, color[0], color[1], color[2])
            os.system(cmd)
            gCurrentColor = color


def mqttOnPublish(mosq, obj, mid):
# log.debug("mid: "+str(mid)))
    pass


def mqttOnSubscribe(mosq, obj, mid, granted_qos):
    log.debug("Subscribed")


def mqttOnLog(mosq, obj, level, string):
    log.debug("log:"+string)


def mqttThread():
    global gQuitting
    log.info("MQTT thread started")
    try:
        mqttc.loop_forever()
        gQuitting = True
        log.info("MQTT thread exiting")
        gQuitting = True
    except Exception as e:
        log.error("MQTT thread got exception: %s" % (e))
        print(traceback.format_exc())
#        gQuitting = True
#        log.info("MQTT disconnect")
#        mqttc.disconnect();
    log.info("MQTT thread exited")

def mqttConnect():
    global mqttc
    global args
    try:
        # If you want to use a specific client id, use
        # mqttc = mosquitto.Mosquitto("client-id")
        # but note that the client id must be unique on the broker. Leaving the client
        # id parameter empty will generate a random id for you.
        mqttc = mosquitto.Mosquitto("airlinecolor-%d" % (random.randint(0, 65535)))
        mqttc.on_message = mqttOnMessage
        mqttc.on_connect = mqttOnConnect
        mqttc.on_disconnect = mqttOnDisconnect
        mqttc.on_publish = mqttOnPublish
        mqttc.on_subscribe = mqttOnSubscribe

        #mqttc.on_log = mqttOnLog # Uncomment to enable debug messages
        mqttc.connect(args.mqtt_host, args.mqtt_port, 60)
        if 1:
            log.info("MQTT thread started")
            try:
                mqttc.loop_start()
                while True:
                    time.sleep(60)
                log.info("MQTT thread exiting")
            except Exception as e:
                log.error("MQTT thread got exception: %s" % (e))
                print(traceback.format_exc())
        #        gQuitting = True
        #        log.info("MQTT disconnect")
        #        mqttc.disconnect();
            log.info("MQTT thread exited")
        else:
            thread = Thread(target = mqttThread)
            thread.daemon = True
            thread.start()
        return True
    except socket.error as e:
        log.error("Failed to connect MQTT broker at %s:%d" % (args.mqtt_host, args.mqtt_port))
        return False

    log.info("MQTT wierdness")


def loggingInit(level, log_host):
    log = logging.getLogger(__name__)

    # Initialize remote logging
    logger = logging.getLogger()
    logger.setLevel(level)
    if log_host != None:
        remotelogger.init(logger = logger, appName = "airlinecol", subSystem = None, host = log_host, level = logging.DEBUG)

    # Log to stdout
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def signal_handler(signal, frame):
    global gQuitting
    global mqttc
    print("ctrl-c")
    gQuitting = True
    mqttc.disconnect();
    sys.exit(0)

def main():
    global gQuitting
    global mqttc
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mqtt-host', dest='mqtt_host', help="MQTT broker hostname", default='127.0.0.1')
    parser.add_argument('-p', '--mqtt-port', dest='mqtt_port', type=int, help="MQTT broker port number", default=1883)
    parser.add_argument('-t', '--mqtt-topic', dest='mqtt_topic', help="MQTT color topic", default="airlinecolor")
    parser.add_argument('-d', '--max-distance', dest='max_distance', type=float, help="Max distance to light the LED (km)", default=10.0)
    parser.add_argument('-v', '--verbose', dest='verbose', action="store_true", help="Verbose output")
    parser.add_argument('-l', '--logger', dest='log_host', help="Remote log host")

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    imagecolor.loadColorData()

    try:
        signal.signal(signal.SIGINT, signal_handler)
        if args.verbose:
            loggingInit(logging.DEBUG, args.log_host)
        else:
            loggingInit(logging.INFO, args.log_host)
        log.info("Client started")

        mqttConnect()
    except Exception as e:
        log.error("Mainloop got exception: %s" % (e))
        print(traceback.format_exc())
        gQuitting = True
    log.debug("MQTT disconnect")
    mqttc.disconnect();

# Ye ol main
if __name__ == "__main__":
    main()
