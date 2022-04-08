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
import threading

gpsd = None #seting the global variable
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
timeHeartbeat = time.mktime(time.gmtime()) + 10
ID = str(random.randint(1,100001))


class GpsPoller(threading.Thread):
  def __init__(self):
    threading.Thread.__init__(self)
    global gpsd #bring it in scope
    gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)
    self.current_value = None
    self.running = True #setting the thread running to true

  def run(self):
    global gpsd
    while self.running:
      gpsd.next() #this will continue to loop and grab EACH set of gpsd info to clear the buffer

parser = argparse.ArgumentParser(description='An MQTT based camera controller', 
epilog="Example: python3 egi_mqtt.py -l 38.87281961220809 -L -77.03190539736168 -a 20.4")
parser.add_argument('-m', '--mqtt-host', help="MQTT broker hostname", default='127.0.0.1')
parser.add_argument('-l', '--latitude', help="Latitude (decimal degrees)", required=True)
parser.add_argument('-L', '--longitude', help="Longitude (decimal degrees)", required=True)
parser.add_argument('-a', '--altitude', help="Altitude (meters)", required=True)
parser.add_argument('-r', '--roll', help="Roll Angle of Camera (degrees)", default=0)
parser.add_argument('-p', '--pitch', help="Pitch Angle of Camera (degrees)", default=0)
parser.add_argument('-y', '--yaw', help="Yaw Angle of Camera (degrees from True North)", default=0)
try:
    args = parser.parse_args()
except:
    logging.critical("Error in Command Line Argument Parsing.  Are all environment variables set?")
    raise
    

state = {}
state['time'] = defaultTime = time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
state['lat'] = defaultLat = float(args.latitude)
state['long'] = defaultLong = float(args.longitude)
state['alt'] = defaultAlt = float(args.altitude)
state['roll'] = defaultRoll = float(args.roll)
state['pitch'] = defaultPitch = float(args.pitch)
state['yaw'] = defaultYaw = float(args.yaw)
state['fix'] = 0
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
try:
    clientLocal.connect(broker_address) #connect to broker
except:
    logging.critical("Could not connect to MQTT Broker.", exc_info=True)
    raise
clientLocal.loop_start() #start the loop
clientLocal.publish("skyscan/registration","EGI-"+ID+" Registration")

gpsp = GpsPoller() # create the thread
try:
    gpsp.start() # start it up
    #############################################
    ##                Main Loop                ##
    #############################################
    while Active:
        state['fix'] = gpsd.fix.mode
        
        # check for 3D fix and update GPS state
        if state['fix']==3: 
            state['time'] = gpsd.fix.time
            state['lat'] = gpsd.fix.latitude
            state['long'] = gpsd.fix.longitude
            state['alt'] = gpsd.fix.altitude
        else:
            state['time'] = defaultTime
            state['lat'] = defaultLat
            state['long'] = defaultLong
            state['alt'] = defaultAlt
        
        # 10 second mqtt publish interval
        if timeTrigger < time.mktime(time.gmtime()):
            if state['fix'] != 3:
                logging.info("No GPS fix. Using EGI values from .env. Check GPS attached on /dev/ACM0 and clear view of sky.")
            timeTrigger = time.mktime(time.gmtime()) + 10
            clientLocal.publish(local_topic,json.dumps(state))
            
        # 30 second heartbeat interval
        if timeHeartbeat < time.mktime(time.gmtime()):
            timeHeartbeat = time.mktime(time.gmtime()) + 30
            logging.info("EGI Heartbeat - Current EGI State: " + json.dumps(state))
        
        delay = 0.01
        time.sleep(delay)
except (KeyboardInterrupt, SystemExit): #when you press ctrl+c
    logging.info("Killing GPS Thread...")
    gpsp.running = False
    gpsp.join(2)
except:
    logging.critical("Error starting GPS.", exc_info=True)
    gpsp.running = False
    gpsp.join(2)
    raise