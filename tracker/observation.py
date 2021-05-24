from datetime import datetime, timedelta

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
    __distance = None
    __bearing = None
    __elevation = None
    __planedb_nagged = False  # Used in case the icao24 is unknown and we only want to log this once
    __onGround = None

    def __init__(self, sbs1msg):

        self.__icao24 = sbs1msg["icao24"].lower() #lets always keep icao24 in lower case
        self.__loggedDate = datetime.utcnow()  # sbs1msg["loggedDate"]
        self.__callsign = sbs1msg["callsign"]
        self.__altitude = sbs1msg["altitude"]
        self.__altitudeTime = datetime.utcnow()
        self.__groundSpeed = sbs1msg["groundSpeed"]
        self.__track = sbs1msg["track"]
        self.__lat = sbs1msg["lat"]
        self.__lon = sbs1msg["lon"]
        self.__latLonTime = datetime.utcnow()
        self.__verticalRate = sbs1msg["verticalRate"]
        self.__onGround = sbs1msg["onGround"]
        self.__operator = None
        self.__registration = None
        self.__type = None
        self.__model = None
        self.__manufacturer = None
        self.__updated = True
        plane = planes.loc[planes['icao24'] == self.__icao24]
        
        if plane.size == 27: # There are 27 columns in CSV file. If it found the plane, it will have 27 keys
            
            logging.info("{}\t[ADDED]\t\t{} {} {} {} {}".format(self.__icao24.lower(), plane["registration"].values[0],plane["manufacturername"].values[0], plane["model"].values[0], plane["operator"].values[0], plane["owner"].values[0]))

            self.__registration = plane['registration'].values[0]
            self.__type = str(plane['manufacturername'].values[0]) + " " + str(plane['model'].values[0])
            self.__manufacturer = plane['manufacturername'].values[0] 
            self.__model =  plane['model'].values[0] 
            self.__operator = plane['operator'].values[0] 
        else:
            if not self.__planedb_nagged:
                self.__planedb_nagged = True
                logging.error("%s\t Not found in the database" % (self.__icao24))
                

    
    def update(self, sbs1msg):
        """ Updates information about a plane from an SBS1 message """

        oldData = dict(self.__dict__) # save existing data to determine if anything has changed
        self.__loggedDate = datetime.utcnow()

        if sbs1msg["icao24"]:
            self.__icao24 = sbs1msg["icao24"].lower() # Let's always keep icao24 in lower case
        if sbs1msg["callsign"] and self.__callsign != sbs1msg["callsign"]:
            self.__callsign = sbs1msg["callsign"].rstrip()
        if sbs1msg["altitude"] is not None:
            self.__altitude = sbs1msg["altitude"]
            self.__altitudeTime = sbs1msg["generatedDate"]
        if sbs1msg["groundSpeed"] is not None:
            self.__groundSpeed = sbs1msg["groundSpeed"]
        if sbs1msg["track"] is not None:
            self.__track = sbs1msg["track"]
        if sbs1msg["onGround"] is not None:
            self.__onGround = sbs1msg["onGround"]
        if sbs1msg["lat"] is not None:
            self.__lat = sbs1msg["lat"]
            self.__latLonTime = sbs1msg["generatedDate"]
        if sbs1msg["lon"] is not None:
            self.__lon = sbs1msg["lon"]
            self.__latLonTime = sbs1msg["generatedDate"]
        if sbs1msg["verticalRate"] is not None:
            self.__verticalRate =  sbs1msg["verticalRate"]

        if not self.__verticalRate:
            self.__verticalRate = 0

 
        if self.__lat and self.__lon and self.__altitude and self.__track:
            # Calculates the distance from the cameras location to the airplane. The output is in METERS!
            distance3d = utils.coordinate_distance_3d(camera_latitude, camera_longitude, camera_altitude, self.__lat, self.__lon, self.__altitude)
            distance2d = utils.coordinate_distance(camera_latitude, camera_longitude,  self.__lat, self.__lon )
            

            self.__distance = distance3d  
            self.__bearing = utils.bearingFromCoordinate(cameraPosition=[camera_latitude, camera_longitude], airplanePosition=[self.__lat, self.__lon], heading=self.__track)
            self.__elevation = utils.elevation(distance2d, cameraAltitude=camera_altitude, airplaneAltitude=self.__altitude) # Distance and Altitude are both in meters
        
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

    def getDistance(self) -> int:
        return self.__distance

    def getLoggedDate(self) -> datetime:
        return self.__loggedDate

    def getLatLonTime(self) -> datetime:
        return self.__latLonTime
    
    def getAltitudeTime(self) -> datetime:
        return self.__altitudeTime

    def getGroundSpeed(self) -> float:
        return self.__groundSpeed

    def getTrack(self) -> float:
        return self.__track

    def getOnGround(self) -> bool:
        return self.__onGround

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

    def isPresentable(self) -> bool:
        return self.__altitude and self.__groundSpeed and self.__track and self.__lat and self.__lon 

    def dump(self):
        """Dump this observation on the console
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logging.debug("> %s  %s %-7s - trk:%3d spd:%3d alt:%5d (%5d) %.4f, %.4f" % (now, self.__icao24, self.__callsign, self.__track, self.__groundSpeed, self.__altitude, self.__verticalRate, self.__lat, self.__lon))

    def json(self, bearing: float, elevation: float, cameraPan: float,  cameraTilt: float, distance: int) -> str:
        """Return JSON representation of this observation
        
        Arguments:
            bearing {float} -- bearing to observation in degrees
            distance {int} -- distance to observation in meters
        
        Returns:
            str -- JSON string
        """

        if self.__callsign is None:
            callsign = "None"
        else:
            callsign = "\"%s\"" % self.__callsign

        planeDict = {"verticalRate": self.__verticalRate, "time": time.time(), "lat": self.__lat, "lon": self.__lon,  "altitude": self.__altitude, "groundSpeed": self.__groundSpeed, "icao24": self.__icao24, "registration": self.__registration, "track": self.__track, "operator": self.__operator,   "loggedDate": self.__loggedDate, "type": self.__type, "manufacturer": self.__manufacturer, "model": self.__model, "callsign": callsign, "bearing": bearing, "cameraPan": cameraPan, "distance": distance, "elevation": elevation, "cameraTilt": cameraTilt}
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
        return d
