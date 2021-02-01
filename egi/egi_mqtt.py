#!/usr/bin/env python3
 
from gps import *
import paho.mqtt.client as mqtt #import the client1
import time
import random
import json
import os
import argparse
import logging
import coloredlogs

Active = True
styles = {'critical': {'bold': True, 'color': 'red'}, 'debug': {'color': 'green'}, 'error': {'color': 'red'}, 'info': {'color': 'white'}, 'notice': {'color': 'magenta'}, 'spam': {'color': 'green', 'faint': True}, 'success': {'bold': True, 'color': 'green'}, 'verbose': {'color': 'blue'}, 'warning': {'color': 'yellow'}}
level = logging.INFO 
coloredlogs.install(level=level, fmt='%(asctime)s.%(msecs)03d \033[0;90m%(levelname)-8s '
                    ''
                    '\033[0;36m%(filename)-18s%(lineno)3d\033[00m '
                    '%(message)s',
                    level_styles = styles)
logging.info("Initializing EGI")

#######################################################
##                Initialize Variables               ##
#######################################################
config = {}
config['Local'] = ["127.0.0.1", "skyscan/egi", "Local MQTT Bus"]  # updated based on naming convention here: https://www.hivemq.com/blog/mqtt-essentials-part-5-mqtt-topics-best-practices/
timeTrigger = time.mktime(time.gmtime()) + 10
timeHeartbeat = 0
ID = str(random.randint(1,100001))
gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE) 
LLA = [38.9510808,-77.3841834,130.1337]
RPY = [0,0,0]

parser = argparse.ArgumentParser(description='An MQTT based camera controller')
parser.add_argument('-m', '--mqtt-host', help="MQTT broker hostname", default='127.0.0.1')
parser.add_argument('-l', '--latitude', help="Latitude (decimal degrees)", default=LLA[0])
parser.add_argument('-L', '--longitude', help="Longitude (decimal degrees)", default=LLA[1])
parser.add_argument('-a', '--altitude', help="Altitude (meters)", default=LLA[2])
parser.add_argument('-r', '--roll', help="Roll Angle of Camera (degrees)", default=RPY[0])
parser.add_argument('-p', '--pitch', help="Pitch Angle of Camera (degrees)", default=RPY[1])
parser.add_argument('-y', '--yaw', help="Yaw Angle of Camera (degrees from True North)", default=RPY[2])
args = parser.parse_args()

state = {}
state['time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
state['lat'] = float(args.latitude)
state['long'] = float(args.longitude)
state['alt'] = float(args.altitude)
state['roll'] = float(args.roll)
state['pitch'] = float(args.pitch)
state['yaw'] = float(args.yaw)
logging.info("Initial State Array: " + str(state))


#######################################################
##           Local MQTT Callback Function            ##
#######################################################
def on_message_local(client, userdata, message):
    payload = str(message.payload.decode("utf-8"))
    logging.info('Message Received: ' + message.topic + ' | ' + payload)
    
def on_disconnect(client, userdata, rc):
    global Active
    Active = False

#############################################
##       Initialize Local MQTT Bus         ##
#############################################
Unit = 'Local'
broker_address=config[Unit][0]
broker_address=args.mqtt_host
local_topic= config[Unit][1]
logging.info("connecting to MQTT broker at "+broker_address+", channel '"+local_topic+"'")
clientLocal = mqtt.Client("EGI-"+ID) #create new instance
clientLocal.on_message = on_message_local #attach function to callback
clientLocal.on_disconnect = on_disconnect
clientLocal.connect(broker_address) #connect to broker
clientLocal.loop_start() #start the loop
clientLocal.publish("skyscan/registration","EGI-"+ID+" Registration")

#############################################
##                Main Loop                ##
#############################################
while Active:
    if timeTrigger < time.mktime(time.gmtime()):
        timeTrigger = time.mktime(time.gmtime()) + 10
        clientLocal.publish(local_topic,json.dumps(state))

    report = gpsd.next() #
    if report['class'] == 'TPV':
        state['time'] = getattr(report,'time',time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
        state['lat'] = getattr(report,'lat',LLA[0])
        state['long'] = getattr(report,'lon',LLA[1])
        state['alt'] = getattr(report,'alt',LLA[2])
    time.sleep(0.01)