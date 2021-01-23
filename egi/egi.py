#!/usr/bin/env python3
 
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

#######################################################
##                Initialize Variables               ##
#######################################################
config = {}
config['Local'] = ["127.0.0.1", "skyscan/egi", "Local MQTT Bus"]  # updated based on naming convention here: https://www.hivemq.com/blog/mqtt-essentials-part-5-mqtt-topics-best-practices/
timeTrigger = 0
timeHeartbeat = 0
ID = str(random.randint(1,100001))

LLA = [ os.environ['LAT'], os.environ['LONG'], os.environ['ALT'] ]
RPY = [ os.environ['ROLL'], os.environ['PITCH'], os.environ['YAW'] ]

state = {}
state['lat'] = LLA[0]
state['long'] = LLA[1]
state['alt'] = LLA[2]
state['roll'] = RPY[0]
state['pitch'] = RPY[1]
state['yaw'] = RPY[2]
state=json.dumps(state)

parser = argparse.ArgumentParser(description='An MQTT based camera controller')

parser.add_argument('-m', '--mqtt-host', help="MQTT broker hostname", default='127.0.0.1')

args = parser.parse_args()


#######################################################
##           Local MQTT Callback Function            ##
#######################################################
def on_message_local(client, userdata, message):
    payload = str(message.payload.decode("utf-8"))
    logging.info('Message Received: ' + message.topic + ' | ' + payload)
    #if message.topic == local_topic+"/OFF":
    #    logging.info("Turning Lamp OFF")
    
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
#clientLocal.subscribe(local_topic+"/#") #config/#")
clientLocal.publish(local_topic+"/registration","EGI-"+ID+" Registration")

#############################################
##                Main Loop                ##
#############################################
while Active:
    #if timeHeartbeat < time.mktime(time.gmtime()):
    #    timeHeartbeat = time.mktime(time.gmtime()) + 10
    #    clientLocal.publish(local_topic+"Heartbeat","EGI-"+ID+" Heartbeat")
    if timeTrigger < time.mktime(time.gmtime()):
        timeTrigger = time.mktime(time.gmtime()) + 10
        clientLocal.publish(local_topic,state)
    time.sleep(0.1)