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
        self.__icao24 = sbs1msg["icao24"]
        self.__loggedDate = datetime.utcnow()  # sbs1msg["loggedDate"]
        self.__callsign = sbs1msg["callsign"]
        self.__altitude = 0.3048 * float(sbs1msg["altitude"] or 0)  # Altitude is in FEET, we convert it to METER. 
        self.__altitudeTime = datetime.utcnow()
        self.__groundSpeed = sbs1msg["groundSpeed"]
        self.__track = sbs1msg["track"]
        self.__lat = sbs1msg["lat"]
        self.__lon = sbs1msg["lon"]
        self.__latLonTime = datetime.utcnow()
        self.__verticalRate = sbs1msg["verticalRate"]
        self.__operator = None
        self.__registration = None
        self.__type = None
        self.__model = None
        self.__manufacturer = None
        self.__updated = True
        plane = planes.loc[planes['icao24'] == self.__icao24.lower()]
        
        if plane.size == 27:
            logging.info("{} {} {} {} {}".format(plane["registration"].values[0],plane["manufacturername"].values[0], plane["model"].values[0], plane["operator"].values[0], plane["owner"].values[0]))

            self.__registration = plane['registration'].values[0]
            self.__type = str(plane['manufacturername'].values[0]) + " " + str(plane['model'].values[0])
            self.__manufacturer = plane['manufacturername'].values[0] 
            self.__model =  plane['model'].values[0] 
            self.__operator = plane['operator'].values[0] 
        else:
            if not self.__planedb_nagged:
                self.__planedb_nagged = True
                logging.error("icao24 %s not found in the database" % (self.__icao24))
                logging.error(plane)

    
    def update(self, sbs1msg):
        oldData = dict(self.__dict__)
        self.__loggedDate = datetime.utcnow()
        if sbs1msg["icao24"]:
            self.__icao24 = sbs1msg["icao24"]
        if sbs1msg["callsign"] and self.__callsign != sbs1msg["callsign"]:
            self.__callsign = sbs1msg["callsign"].rstrip()
        if sbs1msg["altitude"]:
            self.__altitude = 0.3048 * float(sbs1msg["altitude"] or 0)  # Altitude is in FEET, we convert it to METER. 
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
        #if sbs1msg["loggedDate"]:
        #    self.__loggedDate = datetime.strptime(sbs1msg["loggedDate"], '%Y-%m-%d %H:%M:%S.%f')
        #if sbs1msg["generatedDate"]:
        #    self.__generatedDate = sbs1msg["generatedDate"]
        #if sbs1msg["loggedDate"]:
        #    self.__loggedDate = sbs1msg["loggedDate"]

        if self.__planedb_nagged == False and self.__registration == None:
            plane = planes.loc[planes['icao24'] == self.__icao24.lower()]
            
            if plane.size == 27:
                logging.info("{} {} {} {} {}".format(plane["registration"].values[0],plane["manufacturername"].values[0], plane["model"].values[0], plane["operator"].values[0], plane["owner"].values[0]))

                self.__registration = plane['registration'].values[0]
                self.__type = str(plane['manufacturername'].values[0]) + " " + str(plane['model'].values[0])
                self.__manufacturer = plane['manufacturername'].values[0] 
                self.__model =  plane['model'].values[0] 
                self.__operator = plane['operator'].values[0] 
            else:
                if not self.__planedb_nagged:
                    self.__planedb_nagged = True
                    logging.error("icao24 %s not found in the database" % (self.__icao24))
        if self.__lat and self.__lon and self.__altitude:
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
    def getLatLonTime(self) -> datetime:
        return self.__latLonTime
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

def update_config(config):
    global camera_lead
    global min_elevation

    if "cameraLead" in config:
        camera_lead = float(config["cameraLead"])
        logging.info("Setting Camera Lead to: {}".format(camera_lead))
    if "minElevation" in config:
        min_elevation = int(config["minElevation"])
        logging.info("Setting Min. Elevation to: {}".format(min_elevation))

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

                (lat, lon) = utils.calc_travel(cur.getLat(), cur.getLon(), cur.getLatLonTime(), cur.getGroundSpeed(), cur.getTrack(), camera_lead)
                distance = utils.coordinate_distance(camera_latitude, camera_longitude, lat, lon)
                # Round off to nearest 100 meters
                #distance = round(distance/100) * 100
                bearing = utils.bearing(camera_latitude, camera_longitude, lat, lon)
                elevation = utils.elevation(distance, cur.getAltitude(), camera_altitude) # we need to convert to feet because the altitude is in feet

                retain = False
                self.__client.publish(self.__flight_topic, cur.json(bearing, distance, elevation), 0, retain)
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

        
        while True:
            if not self.dump1090Connect():
                continue
            for data in self.dump1090Read():
                if data is None:
                    continue
                self.cleanObservations()
                m = sbs1.parse(data)
                if m:
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
            time.sleep(0.01)
                              
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
    planes = pd.read_csv("/app/data/aircraftDatabase.csv") #,index_col='icao24')
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
