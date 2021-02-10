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

ID = str(random.randint(1,100001))

# Clean out observations this often
OBSERVATION_CLEAN_INTERVAL = 30
q=Queue() # Good writeup of how to pass messages from MQTT into classes, here: http://www.steves-internet-guide.com/mqtt-python-callbacks/
args = None
camera_latitude = None
plant_topic = None # the onMessage function needs to be outside the Class and it needs to get the Plane Topic, so it prob needs to be a global
camera_longitude = None
camera_altitude = None
camera_lead = None
min_elevation = None

# http://stackoverflow.com/questions/1165352/fast-comparison-between-two-python-dictionary
class DictDiffer(object):
    """
    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values
    """
    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.set_current, self.set_past = set(current_dict.keys()), set(past_dict.keys())
        self.intersect = self.set_current.intersection(self.set_past)

    def added(self):
        return self.set_current - self.intersect

    def removed(self):
        return self.set_past - self.intersect

    def changed(self):
        return set(o for o in self.intersect if self.past_dict[o] != self.current_dict[o])

    def unchanged(self):
        return set(o for o in self.intersect if self.past_dict[o] == self.current_dict[o])


class Observation(object):
    """
    This class keeps track of the observed flights around us.
    """
    __icao24 = None
    __loggedDate = None
    __callsign = None
    __altitude = None
    __altitudeTime = None
    __groundSpeed = None
    __track = None
    __lat = None
    __lon = None
    __latLonTime = None
    __verticalRate = None
    __operator = None
    __registration = None
    __type = None
    __manufacturer = None
    __model = None
    __updated = None
    __route = None
    __image_url = None
    __distance = None
    __bearing = None
    __elevation = None
    __planedb_nagged = False  # Used in case the icao24 is unknown and we only want to log this once

    def __init__(self, sbs1msg):
        logging.info("%s appeared" % sbs1msg["icao24"])
    
    def update(self, sbs1msg):
        oldData = dict(self.__dict__)
        #self.__loggedDate = datetime.utcnow()
        if sbs1msg["icao24"]:
            self.__icao24 = sbs1msg["icao24"]
        if sbs1msg["callsign"] and self.__callsign != sbs1msg["callsign"]:
            self.__callsign = sbs1msg["callsign"].rstrip()
        if sbs1msg["altitude"]:
            self.__altitude = sbs1msg["altitude"]
            self.__altitudeTime = datetime.utcnow()
        if sbs1msg["groundSpeed"]:
            self.__groundSpeed = sbs1msg["groundSpeed"]
        if sbs1msg["track"]:
            self.__track = sbs1msg["track"]
        if sbs1msg["lat"]:
            self.__lat = sbs1msg["lat"]
            self.__latLonTime = datetime.utcnow()
        if sbs1msg["lon"]:
            self.__lon = sbs1msg["lon"]
            self.__latLonTime = datetime.utcnow()
        if sbs1msg["verticalRate"]:
            self.__verticalRate = sbs1msg["verticalRate"]
        if not self.__verticalRate:
            self.__verticalRate = 0
        if sbs1msg["loggedDate"]:
            self.__loggedDate = datetime.strptime(sbs1msg["loggedDate"], '%Y-%m-%d %H:%M:%S.%f')
        if sbs1msg["registration"]:
            self.__registration = sbs1msg["registration"]
        if sbs1msg["manufacturer"]:
            self.__manufacturer = sbs1msg["manufacturer"]
        if sbs1msg["model"]:
            self.__model = sbs1msg["model"]
        if sbs1msg["operator"]:
            self.__operator = sbs1msg["operator"]       
        if sbs1msg["type"]:
            self.__type = sbs1msg["type"]   

        # Calculates the distance from the cameras location to the airplane. The output is in METERS!
        distance = utils.coordinate_distance(camera_latitude, camera_longitude, self.__lat, self.__lon)
        #Not sure we want to... commented out for now -> Round off to nearest 100 meters
        self.__distance = distance = distance #round(distance/100) * 100
        self.__bearing = utils.bearing(camera_latitude, camera_longitude, self.__lat, self.__lon)
        self.__elevation = utils.elevation(distance, self.__altitude, camera_altitude) # Distance and Altitude are both in meters

        # Check if observation was updated
        newData = dict(self.__dict__)
        #del oldData["_Observation__loggedDate"]
        #del newData["_Observation__loggedDate"]
        d = DictDiffer(oldData, newData)
        self.__updated = len(d.changed()) > 0

    def getIcao24(self) -> str:
        return self.__icao24

    def getLat(self) -> float:
        return self.__lat

    def getLon(self) -> float:
        return self.__lon

    def isUpdated(self) -> bool:
        return self.__updated

    def getElevation(self) -> int:
        return self.__elevation

    def getLoggedDate(self) -> datetime:
        return self.__loggedDate

    def getGroundSpeed(self) -> float:
        return self.__groundSpeed

    def getTrack(self) -> float:
        return self.__track

    def getAltitude(self) -> float:
        return self.__altitude

    def getType(self) -> str:
        return self.__type

    def getManufacturer(self) -> str:
        return self.__manufacturer

    def getModel(self) -> str:
        return self.__model

    def getRegistration(self) -> str:
        return self.__registration

    def getOperator(self) -> str:
        return self.__operator

    def getRoute(self) -> str:
        return self.__route
    
    def getVerticalRate(self) -> float:
        return self.__verticalRate

    def getImageUrl(self) -> str:
        return self.__image_url

    def isPresentable(self) -> bool:
        return self.__altitude and self.__groundSpeed and self.__track and self.__lat and self.__lon # and self.__operator and self.__registration and self.__image_url

    def dump(self):
        """Dump this observation on the console
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logging.debug("> %s  %s %-7s - trk:%3d spd:%3d alt:%5d (%5d) %.4f, %.4f" % (now, self.__icao24, self.__callsign, self.__track, self.__groundSpeed, self.__altitude, self.__verticalRate, self.__lat, self.__lon))

    def json(self, bearing: int, distance: int, elevation: int) -> str:
        """Return JSON representation of this observation
        
        Arguments:
            bearing {int} -- bearing to observation in degrees
            distance {int} -- distance to observation in meters
        
        Returns:
            str -- JSON string
        """
        if self.__route is None:
            route = "None"
        else:
            route = "%s" % self.__route
            route = route.replace("'", "\"")

        if self.__callsign is None:
            callsign = "None"
        else:
            callsign = "\"%s\"" % self.__callsign

        planeDict = {"verticalRate": self.__verticalRate, "time": time.time(), "lat": self.__lat, "lon": self.__lon,  "altitude": self.__altitude, "groundSpeed": self.__groundSpeed, "icao24": self.__icao24, "registration": self.__registration, "track": self.__track, "operator": self.__operator,   "loggedDate": self.__loggedDate, "type": self.__type, "manufacturer": self.__manufacturer, "model": self.__model, "callsign": callsign, "bearing": bearing, "distance": distance, "elevation": elevation}
        jsonString = json.dumps(planeDict, indent=4, sort_keys=True, default=str)
        return jsonString

    def dict(self):
        d =  dict(self.__dict__)
        if d["_Observation__verticalRate"] == None:
            d["verticalRate"] = 0
        if "_Observation__lastAlt" in d:
            del d["lastAlt"]
        if "_Observation__lastLat" in d:
            del d["lastLat"]
        if "_Observation__lastLon" in d:
            del d["lastLon"]
        #d["loggedDate"] = "%s" % (d["_Observation__loggedDate"])
        return d



def on_message(client, userdata, message):
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
    if message.topic == plane_topic:
        q.put(update) #put messages on queue
    elif message.topic == "skyscan/egi":
        logging.info(update)
        camera_longitude = float(update["long"])
        camera_latitude = float(update["lat"])
        camera_altitude = float(update["alt"])
    else:
        logging.info("Topic not processed: " + message.topic)
   
class FlightTracker(object):
    __mqtt_broker: str = ""
    __mqtt_port: int = 0
    __plane_topic: str = None
    __tracking_topic: str = None
    __client = None
    __observations: Dict[str, str] = {}
    __tracking_icao24: str = None
    __tracking_distance: int = 999999999
    __next_clean: datetime = None
    __has_nagged: bool = False

    def __init__(self,  mqtt_broker: str, plane_topic: str, tracking_topic: str, mqtt_port: int = 1883, ):
        """Initialize the flight tracker

        Arguments:
            dump1090_host {str} -- Name or IP of dump1090 host
            mqtt_broker {str} -- Name or IP of dump1090 MQTT broker
            latitude {float} -- Latitude of receiver
            longitude {float} -- Longitude of receiver
            plane_topic {str} -- MQTT topic for plane reports
            tracking_topic {str} -- MQTT topic for current tracking report

        Keyword Arguments:
            dump1090_port {int} -- Override the dump1090 raw port (default: {30003})
            mqtt_port {int} -- Override the MQTT default port (default: {1883})
        """

        self.__mqtt_broker = mqtt_broker
        self.__mqtt_port = mqtt_port
        self.__sock = None
        self.__observations = {}
        self.__next_clean = datetime.utcnow() + timedelta(seconds=OBSERVATION_CLEAN_INTERVAL)
        self.__plane_topic = plane_topic
        self.__tracking_topic = tracking_topic


    def __publish_thread(self):
        """
        MQTT publish closest observation every second, more often if the plane is closer
        """
        timeHeartbeat = 0
        while True:
            if timeHeartbeat < time.mktime(time.gmtime()):
                timeHeartbeat = time.mktime(time.gmtime()) + 10
                self.__client.publish("skyscan/heartbeat", "skyscan-tracker-" +ID+" Heartbeat", 0, False)
            if not self.__tracking_icao24:
                time.sleep(1)
            else:
                if not self.__tracking_icao24 in self.__observations:
                    self.__tracking_icao24 is None
                    continue
                cur = self.__observations[self.__tracking_icao24]
                if cur is None:
                    continue

                (lat, lon) = utils.calc_travel(cur.getLat(), cur.getLon(), cur.getLoggedDate(), cur.getGroundSpeed(), cur.getTrack(), camera_lead)
                distance = utils.coordinate_distance(camera_latitude, camera_longitude, lat, lon)
                # Round off to nearest 100 meters
                #distance = round(distance/100) * 100
                bearing = utils.bearing(camera_latitude, camera_longitude, lat, lon)
                elevation = utils.elevation(distance, cur.getAltitude(), camera_altitude) # we need to convert to feet because the altitude is in feet

                retain = False
                self.__client.publish(self.__tracking_topic, cur.json(bearing, distance, elevation), 0, retain)
                logging.info("%s at %5d brg %3d alt %5d trk %3d spd %3d %s" % (cur.getIcao24(), distance, bearing, cur.getAltitude(), cur.getTrack(), cur.getGroundSpeed(), cur.getType()))

                if distance < 3000:
                    time.sleep(0.25)
                elif distance < 6000:
                    time.sleep(0.5)
                else:
                    time.sleep(1)


    def updateTrackingDistance(self):
        """Update distance to aircraft being tracked
        """
        cur = self.__observations[self.__tracking_icao24]
        self.__tracking_distance = utils.coordinate_distance(camera_latitude, camera_longitude, cur.getLat(), cur.getLon())


    def run(self):
        """Run the flight tracker.
        """
        print("connecting to MQTT broker at "+ self.__mqtt_broker +", subcribing on channel '"+ self.__plane_topic+"'publising on: " + self.__tracking_topic)
        self.__client = mqtt.Client("skyscan-tracker-" + ID) #create new instance

        self.__client.on_message = on_message #attach function to callback
        print("setup MQTT")
        self.__client.connect(self.__mqtt_broker) #connect to broker
        print("connected mqtt")
        self.__client.loop_start() #start the loop
        print("start MQTT")
        self.__client.subscribe(self.__plane_topic)
        self.__client.subscribe("skyscan/egi")
        self.__client.publish("skyscan/registration", "skyscan-tracker-"+ID+" Registration", 0, False)
        print("subscribe mqtt")
        threading.Thread(target = self.__publish_thread, daemon = True).start()

        
        while True:
            while not q.empty():
                m = q.get()
                icao24 = m["icao24"]
                self.cleanObservations()
                if icao24 not in self.__observations:
                    self.__observations[icao24] = Observation(m)
                self.__observations[icao24].update(m)
                if self.__observations[icao24].isPresentable():
                    if not self.__tracking_icao24:
                        if self.__observations[icao24].getElevation() != None and self.__observations[icao24].getElevation() > min_elevation:
                            self.__tracking_icao24 = icao24
                            self.updateTrackingDistance()
                            logging.info("Tracking %s at %d elevation: %d" % (self.__tracking_icao24, self.__tracking_distance, self.__observations[icao24].getElevation()))
                    elif self.__tracking_icao24 == icao24:
                        self.updateTrackingDistance()
                    else:
                        distance = utils.coordinate_distance(camera_latitude, camera_longitude, self.__observations[icao24].getLat(), self.__observations[icao24].getLon())
                        if distance < self.__tracking_distance and self.__observations[icao24].getElevation() > min_elevation:
                            self.__tracking_icao24 = icao24
                            self.__tracking_distance = distance
                            logging.info("Tracking %s at %d elevation: %d" % (self.__tracking_icao24, self.__tracking_distance, self.__observations[icao24].getElevation()))               
            time.sleep(0.1)
                              
    def selectNearestObservation(self):
        """Select nearest presentable aircraft
        """
        self.__tracking_icao24 = None
        self.__tracking_distance = 999999999
        for icao24 in self.__observations:
            if not self.__observations[icao24].isPresentable():
                continue
            if self.__observations[icao24].getElevation() < min_elevation:
                continue
            distance = utils.coordinate_distance(camera_latitude, camera_longitude, self.__observations[icao24].getLat(), self.__observations[icao24].getLon())
            if distance < self.__tracking_distance:
                self.__tracking_icao24 = icao24
                self.__tracking_distance = distance
        if self.__tracking_icao24 is None:
            logging.info("Found nothing to track")
        else:
            logging.info("Found new tracking %s at %d" % (self.__tracking_icao24, self.__tracking_distance))


    def cleanObservations(self):
        """Clean observations for planes not seen in a while
        """
        now = datetime.utcnow()
        if now > self.__next_clean:
            cleaned = []
            for icao24 in self.__observations:
#                logging.info("[%s] %s -> %s : %s" % (icao24, self.__observations[icao24].getLoggedDate(), self.__observations[icao24].getLoggedDate() + timedelta(seconds=OBSERVATION_CLEAN_INTERVAL), now))
                if self.__observations[icao24].getLoggedDate() + timedelta(seconds=OBSERVATION_CLEAN_INTERVAL) < now:
                    logging.info("%s disappeared" % (icao24))
                    if icao24 == self.__tracking_icao24:
                        self.__tracking_icao24 = None
                    cleaned.append(icao24)
                if icao24 == self.__tracking_icao24 and self.__observations[icao24].getElevation() < min_elevation:
                    logging.info("%s is too low to track" % (icao24))
                    self.__tracking_icao24 = None
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
    parser = argparse.ArgumentParser(description='A Dump 1090 to MQTT bridge')


    parser.add_argument('-l', '--lat', type=float, help="Latitude of camera")
    parser.add_argument('-L', '--lon', type=float, help="Longitude of camera")
    parser.add_argument('-a', '--alt', type=float, help="altitude of camera in METERS!", default=0)
    parser.add_argument('-c', '--camera-lead', type=float, help="how many seconds ahead of a plane's predicted location should the camera be positioned", default=0.25)
    parser.add_argument('-M', '--min-elevation', type=int, help="minimum elevation for camera", default=0)
    parser.add_argument('-m', '--mqtt-host', help="MQTT broker hostname", default='127.0.0.1')
    parser.add_argument('-p', '--mqtt-port', type=int, help="MQTT broker port number (default 1883)", default=1883)
    parser.add_argument('-P', '--plane-topic', dest='plane_topic', help="MQTT plane topic", default="skyscan/planes/json")
    parser.add_argument('-T', '--tracking-topic', dest='tracking_topic', help="MQTT tracking topic", default="skyscan/tracking/json")
    parser.add_argument('-v', '--verbose',  action="store_true", help="Verbose output")

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


    tracker = FlightTracker( args.mqtt_host, args.plane_topic, args.tracking_topic,  mqtt_port = args.mqtt_port)
    tracker.run()  # Never returns


# Ye ol main
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e, exc_info=True)
