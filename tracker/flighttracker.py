#!/usr/bin/env python3
#
# Copyright (c) 2020 Johan Kanflo (github.com/kanflo)
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

from typing import *
import socket, select
import argparse
import threading
import json
import sys
import os
import logging
import coloredlogs
import calendar
from datetime import datetime, timedelta
import signal
import random
import time
import re
import errno
import sbs1
import utils
import paho.mqtt.client as mqtt 
from json.decoder import JSONDecodeError
import pandas as pd
from queue import Queue
from observation import Observation

ID = str(random.randint(1,100001))

# Clean out observations this often
OBSERVATION_CLEAN_INTERVAL = 30
# Socket read timeout
DUMP1090_SOCKET_TIMEOUT = 60
q=Queue() # Good writeup of how to pass messages from MQTT into classes, here: http://www.steves-internet-guide.com/mqtt-python-callbacks/
args = None
camera_latitude = None
plant_topic = None # the onMessage function needs to be outside the Class and it needs to get the Plane Topic, so it prob needs to be a global
config_topic = "skyscan/config/json"
camera_longitude = None
camera_altitude = None
camera_lead = None
min_elevation = None
min_altitude = None
max_altitude = None
min_distance = None
max_distance = None


def whyTrackable(observation) -> str:
    """ Returns a string explaining why a Plane can or cannot be tracked """

    reason = ""

    if observation.getAltitude() == None or observation.getGroundSpeed() == None or observation.getTrack() == None or observation.getLat() == None or observation.getLon() == None:
        reason = "Loc: ⛔️" 
    else:
        reason = "Loc: ✅" 

    if observation.getOnGround() == True:
        reason = reason + "\tGrnd: ⛔️" 
    else:
        reason = reason + "\tGrnd: ✅" 
    
    if max_altitude != None and observation.getAltitude() > max_altitude:
        reason = reason + "\tMax Alt: ⛔️" 
    else:
        reason = reason + "\tMax Alt: ✅" 

    if min_altitude != None and observation.getAltitude() < min_altitude:
        reason = reason + "\tMin Alt: ⛔️" 
    else:
        reason = reason + "\tMin Alt: ✅" 

    if observation.getDistance() == None or observation.getElevation() == None:
        return False

    if min_distance != None and observation.getDistance() < min_distance:
        reason = reason + "\tMin Dist: ⛔️" 
    else:
        reason = reason + "\tMin Dist: ✅" 

    if max_distance != None and observation.getDistance() > max_distance:
        reason = reason + "\tMax Dist: ⛔️" 
    else:
        reason = reason + "\tMax Dist: ✅" 
    
    if observation.getElevation() < min_elevation:
        reason = reason + "\tMin Elv: ⛔️" 
    else:
        reason = reason + "\tMin Elv: ✅" 

    return reason

def isTrackable(observation) -> bool:
    """ Does this observation meet all of the requirements to be tracked """

    if observation.getAltitude() == None or observation.getGroundSpeed() == None or observation.getTrack() == None or observation.getLat() == None or observation.getLon() == None:
        return False 

    if observation.getOnGround() == True:
        return False
    
    if max_altitude != None and observation.getAltitude() > max_altitude:
        return False

    if min_altitude != None and observation.getAltitude() < min_altitude:
        return False

    if observation.getDistance() == None or observation.getElevation() == None:
        return False

    if min_distance != None and observation.getDistance() < min_distance:
        return False

    if max_distance != None and observation.getDistance() > max_distance:
        return False
    
    if observation.getElevation() < min_elevation:
        return False

    return True


def update_config(config):
    """ Adjust configuration values based on MQTT config messages that come in """
    global camera_lead
    global min_elevation
    global min_distance
    global min_altitude
    global min_elevation
    global max_altitude
    global max_distance


    if "cameraLead" in config:
        camera_lead = float(config["cameraLead"])
        logging.info("Setting Camera Lead to: {}".format(camera_lead))
    if "minElevation" in config:
        min_elevation = int(config["minElevation"])
        logging.info("Setting Min. Elevation to: {}".format(min_elevation))
    if "minDistance" in config:
        min_distance = int(config["minDistance"])
        logging.info("Setting Min. Distance to: {}".format(min_distance))
    if "minAltitude" in config:
        min_altitude = int(config["minAltitude"])
        logging.info("Setting Min. Altitude to: {}".format(min_altitude))
    if "maxAltitude" in config:
        max_altitude = int(config["maxAltitude"])
        logging.info("Setting Max Altitude to: {}".format(max_altitude))                    
    if "maxDistance" in config:
        max_distance = int(config["maxDistance"])
        logging.info("Setting Max Distance to: {}".format(min_elevation))
        
def on_message(client, userdata, message):
    """ MQTT Client callback for new messages """

    global camera_altitude
    global camera_latitude
    global camera_longitude

    command = str(message.payload.decode("utf-8"))
    # Assumes you will only be getting JSON on your subscribed messages
    try:
        update = json.loads(command)
    except JSONDecodeError as e:
        log.critical("onMessage - JSONDecode Error: {} ".format(e))
    except TypeError as e:
        log.critical("onMessage - Type Error: {} ".format(e))
    except ValueError as e:
        log.critical("onMessage - Value Error: {} ".format(e))
    except:
        log.critical("onMessage - Caught it!")

    if message.topic == "skyscan/egi":
        logging.info(update)
        camera_longitude = float(update["long"])
        camera_latitude = float(update["lat"])
        camera_altitude = float(update["alt"])
    elif message.topic == config_topic:
        update_config(update)
        logging.info("Config Message: {}".format(update))
    else:
        logging.info("Topic not processed: " + message.topic)
   
class FlightTracker(object):
    __mqtt_broker: str = ""
    __mqtt_port: int = 0
    __plane_topic: str = None
    __flight_topic: str = None
    __client = None
    __observations: Dict[str, str] = {}
    __tracking_icao24: str = None
    __tracking_distance: int = 999999999
    __next_clean: datetime = None
    __has_nagged: bool = False
    __dump1090_host: str = ""
    __dump1090_port: int = 0
    __dump1090_sock: socket.socket = None

    def __init__(self, dump1090_host: str, mqtt_broker: str, plane_topic: str, flight_topic: str, dump1090_port: int = 30003, mqtt_port: int = 1883, ):
        """Initialize the flight tracker

        Arguments:
            dump1090_host {str} -- Name or IP of dump1090 host
            mqtt_broker {str} -- Name or IP of dump1090 MQTT broker
            latitude {float} -- Latitude of receiver
            longitude {float} -- Longitude of receiver
            plane_topic {str} -- MQTT topic for plane reports
            flight_topic {str} -- MQTT topic for current tracking report

        Keyword Arguments:
            dump1090_port {int} -- Override the dump1090 raw port (default: {30003})
            mqtt_port {int} -- Override the MQTT default port (default: {1883})
        """
        self.__dump1090_host = dump1090_host
        self.__dump1090_port = dump1090_port
        self.__mqtt_broker = mqtt_broker
        self.__mqtt_port = mqtt_port
        self.__sock = None
        self.__observations = {}
        self.__next_clean = datetime.utcnow() + timedelta(seconds=OBSERVATION_CLEAN_INTERVAL)
        self.__plane_topic = plane_topic
        self.__flight_topic = flight_topic


    def __publish_thread(self):
        """
        MQTT publish closest observation every second, more often if the plane is closer
        """
        timeHeartbeat = 0
        while True:

            # Checks to see if it is time to publish a hearbeat message
            if timeHeartbeat < time.mktime(time.gmtime()):
                timeHeartbeat = time.mktime(time.gmtime()) + 10
                self.__client.publish("skyscan/heartbeat", "skyscan-tracker-" +ID+" Heartbeat", 0, False)

            # if we are not tracking anything, goto sleep for 1 second
            if not self.__tracking_icao24:
                time.sleep(1)
            else:
                if not self.__tracking_icao24 in self.__observations:
                    self.__tracking_icao24 is None
                    continue
                cur = self.__observations[self.__tracking_icao24]
                if cur is None:
                    continue

                (lat, lon, alt) = utils.calc_travel_3d(cur.getLat(), cur.getLon(), cur.getAltitude(), cur.getLatLonTime(), cur.getAltitudeTime(), cur.getGroundSpeed(), cur.getTrack(), cur.getVerticalRate(), camera_lead)
                distance3d = utils.coordinate_distance_3d(camera_latitude, camera_longitude, camera_altitude, lat, lon, alt)
                (latorig, lonorig) = utils.calc_travel(cur.getLat(), cur.getLon(), cur.getLatLonTime(),  cur.getGroundSpeed(), cur.getTrack(), camera_lead)
                distance2d = utils.coordinate_distance(camera_latitude, camera_longitude, lat, lon)

                #logging.info("  ------------------------------------------------- ")
                #logging.info("%s: original alt %5f | extrap alt %5f | climb rate %5f | original climb rate %5f" % (cur.getIcao24(), cur.getAltitude(), alt, cur.getVerticalRate(), cur.getVerticalRate()/0.00508))
                #logging.info("%s: original lat %5f | new lat %5f | original long %5f | new long %5f " % (cur.getIcao24(), latorig, lat, lonorig, lon))
        
                # Round off to nearest 100 meters
                #distance = round(distance/100) * 100
                bearing = utils.bearingFromCoordinate( cameraPosition=[camera_latitude, camera_longitude], airplanePosition=[lat, lon], heading=cur.getTrack())
                elevation = utils.elevation(distance2d, cameraAltitude=camera_altitude, airplaneAltitude=alt) 
                
                # !!!! Mike, replaces these values with the values that have been camera for roll, pitch, yaw
                cameraTilt = elevation
                cameraPan = utils.cameraPanFromCoordinate(cameraPosition=[camera_latitude, camera_longitude], airplanePosition=[lat, lon])
                elevationorig = utils.elevation(distance2d, cur.getAltitude(), camera_altitude) 
                #logging.info("%s: original elevation %5d | new elevation %5d" % (cur.getIcao24(), elevationorig, elevation))

                retain = False
                self.__client.publish(self.__flight_topic, cur.json(bearing=bearing, cameraPan=cameraPan, distance=distance3d, elevation=elevation, cameraTilt=cameraTilt), 0, retain)
                #logging.info("%s at %5d brg %3d alt %5d trk %3d spd %3d %s" % (cur.getIcao24(), distance3d, bearing, cur.getAltitude(), cur.getTrack(), cur.getGroundSpeed(), cur.getType()))
                #logging.info("  ------------------------------------------------- ")
                
                if distance3d < 3000:
                    time.sleep(0.25)
                elif distance3d < 6000:
                    time.sleep(0.5)
                else:
                    time.sleep(1)


    def updateTrackingDistance(self):
        """Update distance to aircraft being tracked
        """
        cur = self.__observations[self.__tracking_icao24]
        if cur.getAltitude():
            self.__tracking_distance = utils.coordinate_distance_3d(camera_latitude, camera_longitude, camera_altitude, cur.getLat(), cur.getLon(), cur.getAltitude())

    def dump1090Connect(self) -> bool:
        """If not connected, connect to the dump1090 host

        Returns:
            bool -- True if we are connected
        """
        if self.__dump1090_sock == None:
            try:
                if not self.__has_nagged:
                    logging.info("Connecting to dump1090")
                self.__dump1090_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.__dump1090_sock.connect((self.__dump1090_host, self.__dump1090_port))
                logging.info("ADSB connected")
                self.__dump1090_sock.settimeout(DUMP1090_SOCKET_TIMEOUT)
                self.__has_nagged = False
                return True
            except socket.error as e:
                if not self.__has_nagged:
                    logging.critical("Failed to connect to ADSB receiver on %s:%s, retrying : %s" % (self.__dump1090_host, self.__dump1090_port, e))
                    self.__has_nagged = True
                self.__dump1090_sock = None
                time.sleep(5)
            return False
        else:
            return True


    def dump1090Close(self):
        """Close connection to dump1090 host.
        """
        try:
            self.__dump1090_sock.close()
        except socket.error:
            pass
        self.__dump1090_sock = None
        self.__has_nagged = False
        logging.critical("Closing dump1090 connection")


    def dump1090Read(self) -> str:
        """Read a line from the dump1090 host. If the host went down, close the socket and return None

        Returns:
            str -- An SBS1 message or None if disconnected or timeout

        Yields:
            str -- An SBS1 message or None if disconnected or timeout
        """
        try:
            try:
                buffer = self.__dump1090_sock.recv(4096)
            except ConnectionResetError:
                logging.critical("Connection Reset Error")
                self.dump1090Close()
                return None
            except socket.error:
                logging.critical("Socket Error")
                self.dump1090Close()
                return None
            buffer = buffer.decode("utf-8")
            buffering = True
            if buffer == "":
                logging.critical("Buffer Empty")
                self.dump1090Close()
                return None
            while buffering:
                if "\n" in buffer:
                    (line, buffer) = buffer.split("\r\n", 1)
                    yield line
                else:
                    try:
                        more = self.__dump1090_sock.recv(4096)
                    except ConnectionResetError:
                        logging.critical("Connection Reset Error")
                        self.dump1090Close()
                        return None
                    except socket.error:
                        logging.critical("Socket Error")
                        self.dump1090Close()
                        return None
                    if not more:
                        buffering = False
                    else:
                        if not isinstance(more, str):
                            more = more.decode("utf-8")
                        if more == "":
                            logging.critical("Receive Empty")
                            self.dump1090Close()
                            return None
                        buffer += more
            if buffer:
                yield buffer
        except socket.timeout:
            return None




    def run(self):
        """Run the flight tracker.
        """
        print("connecting to MQTT broker at "+ self.__mqtt_broker +", subcribing on channel '"+ self.__plane_topic+"'publising on: " + self.__flight_topic)
        self.__client = mqtt.Client("skyscan-tracker-" + ID) #create new instance

        self.__client.on_message = on_message #attach function to callback
        print("setup MQTT")
        self.__client.connect(self.__mqtt_broker) #connect to broker
        print("connected mqtt")
        self.__client.loop_start() #start the loop
        print("start MQTT")
        self.__client.subscribe("skyscan/egi")
        self.__client.subscribe(config_topic)
        self.__client.publish("skyscan/registration", "skyscan-tracker-"+ID+" Registration", 0, False)
        print("subscribe mqtt")
        threading.Thread(target = self.__publish_thread, daemon = True).start()

        # This loop reads in new messages from dump1090 and determines which plane to track
        while True:
            if not self.dump1090Connect():
                continue
            for data in self.dump1090Read():
                if data is None:
                    continue
                self.cleanObservations()
                m = sbs1.parse(data)
                if m:
                    icao24 = m["icao24"].lower()

                    # Add or update the Observation for the plane
                    if icao24 not in self.__observations:
                        self.__observations[icao24] = Observation(m)
                    else:
                        self.__observations[icao24].update(m)

                    # if the plane is suitable to be tracked    
                    if isTrackable(self.__observations[icao24]):

                        # if no plane is being tracked, track this one
                        if not self.__tracking_icao24:
                            self.__tracking_icao24 = icao24
                            self.updateTrackingDistance()
                            logging.info("{}\t[TRACKING]\tDist: {}\tElev: {}\t\t".format(self.__tracking_icao24, self.__tracking_distance, self.__observations[icao24].getElevation()))
          
                        # if this is the plane being tracked, update the tracking distance
                        elif self.__tracking_icao24 == icao24:
                            self.updateTrackingDistance()
                        
                        # This plane is trackable, but is not the one being tracked
                        else:
                            distance = self.__observations[icao24].getDistance()
                            if distance < self.__tracking_distance:
                                self.__tracking_icao24 = icao24
                                self.__tracking_distance = distance
                                logging.info("{}\t[TRACKING]\tDist: {}\tElev: {}\t\t - Switched to closer plane".format(self.__tracking_icao24, int(self.__tracking_distance), int(self.__observations[icao24].getElevation())))
                    else:
                        # If the plane is currently being tracked, but is no longer trackable:
                        if self.__tracking_icao24 == icao24:
                            logging.info("%s\t[NOT TRACKING]\t - Observation is no longer trackable" % (icao24))
                            logging.info(whyTrackable(self.__observations[icao24]))
                            self.__tracking_icao24 = None
                            self.__tracking_distance = 999999999
                                                  
            time.sleep(0.01)

    def selectNearestObservation(self):
        """Select nearest presentable aircraft
        """
        self.__tracking_icao24 = None
        self.__tracking_distance = 999999999
        for icao24 in self.__observations:
            if not isTrackable(self.__observations[icao24]):
                continue
            distance = self.__observations[icao24].getDistance()
            if self.__observations[icao24].getDistance() < self.__tracking_distance:
                self.__tracking_icao24 = icao24
                self.__tracking_distance = distance
        if self.__tracking_icao24:
            logging.info("{}\t[TRACKING]\tDist: {}\t\t - Selected Nearest Observation".format(self.__tracking_icao24, self.__tracking_distance))
            

    def cleanObservations(self):
        """Clean observations for planes not seen in a while
        """
        now = datetime.utcnow()
        if now > self.__next_clean:
            cleaned = []
            for icao24 in self.__observations:
#                logging.info("[%s] %s -> %s : %s" % (icao24, self.__observations[icao24].getLoggedDate(), self.__observations[icao24].getLoggedDate() + timedelta(seconds=OBSERVATION_CLEAN_INTERVAL), now))
                if self.__observations[icao24].getLoggedDate() + timedelta(seconds=OBSERVATION_CLEAN_INTERVAL) < now:
                    logging.info("%s\t[REMOVED]\t" % (icao24))
                    if icao24 == self.__tracking_icao24:
                        self.__tracking_icao24 = None
                        self.__tracking_distance = 999999999
                    cleaned.append(icao24)
                if icao24 == self.__tracking_icao24 and not isTrackable(self.__observations[icao24]):
                    logging.info("%s\t[NOT TRACKING]\t - Observation is no longer trackable" % (icao24))
                    logging.info(whyTrackable(self.__observations[icao24]))
                    self.__tracking_icao24 = None
                    self.__tracking_distance = 999999999
            for icao24 in cleaned:
                del self.__observations[icao24]
            if self.__tracking_icao24 is None:
                self.selectNearestObservation()

            self.__next_clean = now + timedelta(seconds=OBSERVATION_CLEAN_INTERVAL)


def main():
    global args
    global logging
    global camera_altitude
    global camera_latitude
    global camera_longitude
    global camera_lead
    global plane_topic
    global min_elevation
    global planes
    parser = argparse.ArgumentParser(description='A Dump 1090 to MQTT bridge')


    parser.add_argument('-l', '--lat', type=float, help="Latitude of camera")
    parser.add_argument('-L', '--lon', type=float, help="Longitude of camera")
    parser.add_argument('-a', '--alt', type=float, help="altitude of camera in METERS!", default=0)
    parser.add_argument('-c', '--camera-lead', type=float, help="how many seconds ahead of a plane's predicted location should the camera be positioned", default=0.25)
    parser.add_argument('-M', '--min-elevation', type=int, help="minimum elevation for camera", default=0)
    parser.add_argument('-m', '--mqtt-host', help="MQTT broker hostname", default='127.0.0.1')
    parser.add_argument('-p', '--mqtt-port', type=int, help="MQTT broker port number (default 1883)", default=1883)
    parser.add_argument('-P', '--plane-topic', dest='plane_topic', help="MQTT plane topic", default="skyscan/planes/json")
    parser.add_argument('-T', '--flight-topic', dest='flight_topic', help="MQTT flight tracking topic", default="skyscan/flight/json")
    parser.add_argument('-v', '--verbose',  action="store_true", help="Verbose output")
    parser.add_argument('-H', '--dump1090-host', help="dump1090 hostname", default='127.0.0.1')
    parser.add_argument('--dump1090-port', type=int, help="dump1090 port number (default 30003)", default=30003)
 
    args = parser.parse_args()

    if not args.lat and not args.lon:
        log.critical("You really need to tell me where you are located (--lat and --lon)")
        sys.exit(1)
    camera_longitude = args.lon
    camera_latitude = args.lat
    camera_altitude = args.alt # Altitude is in METERS
    plane_topic = args.plane_topic
    camera_lead = args.camera_lead
    min_elevation = args.min_elevation
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
    planes = pd.read_csv("/data/aircraftDatabase.csv") #,index_col='icao24')
    logging.info("Printing table")
    logging.info(planes)

    tracker = FlightTracker(args.dump1090_host, args.mqtt_host, args.plane_topic, args.flight_topic,dump1090_port = args.dump1090_port,  mqtt_port = args.mqtt_port)
    tracker.run()  # Never returns


# Ye ol main
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e, exc_info=True)
