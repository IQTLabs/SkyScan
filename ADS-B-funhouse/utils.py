from typing import *
import logging
import math
import bing
import planedb
from datetime import datetime, timedelta


def deg2rad(deg: float) -> float:
    """Convert degrees to radians

    Arguments:
        deg {float} -- Angle in degrees

    Returns:
        float -- Angle in radians
    """
    return deg * (math.pi/180)

def azimuth(distance: float, altitude):
    baseElevation = 0
    ratio = ( altitude - baseElevation) / distance
    a = math.atan(ratio) * (180 /math.pi)
    return round(a)

def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing from lat1/lon2 to lat2/lon2

    Arguments:
        lat1 {float} -- Start latitude
        lon1 {float} -- Start longitude
        lat2 {float} -- End latitude
        lon2 {float} -- End longitude

    Returns:
        float -- bearing in degrees
    """
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    rlon1 = math.radians(lon1)
    rlon2 = math.radians(lon2)
    dlon = math.radians(lon2-lon1)

    b = math.atan2(math.sin(dlon)*math.cos(rlat2),math.cos(rlat1)*math.sin(rlat2)-math.sin(rlat1)*math.cos(rlat2)*math.cos(dlon)) # bearing calc
    bd = math.degrees(b)
    br,bn = divmod(bd+360,360) # the bearing remainder and final bearing

    return bn


def coordinate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between the two coordinates

    Arguments:
        lat1 {float} -- Start latitude
        lon1 {float} -- Start longitude
        lat2 {float} -- End latitude
        lon2 {float} -- End longitude

    Returns:
        float -- Distance in meters
    """
    R = 6371 # Radius of the earth in km
    dLat = deg2rad(lat2-lat1)
    dLon = deg2rad(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c * 1000 #  Distance in m
    return d


def calc_travel(lat: float, lon: float, utc_start: datetime, speed_kts: float, heading: float) -> Tuple[float, float]:
    """Calculate travel from lat, lon starting at a certain time with giben speed and heading

    Arguments:
        lat {float} -- Starting latitude
        lon {float} -- Starting longitude
        utc_start {datetime} -- Start time
        speed_kts {float} -- Speed in knots
        heading {float} -- Heading in degress

    Returns:
        Tuple[float, float] -- The new lat/lon as a tuple
    """
    age = datetime.utcnow() - utc_start
    age_s = age.total_seconds()

    R = 6378.1 # Radius of the Earth
    brng = math.radians(heading) # Bearing is 90 degrees converted to radians.
    speed_mps = 0.514444 * speed_kts # knots -> m/s
    d = (age_s * speed_mps) / 1000.0 # Distance in km

    lat1 = math.radians(lat) # Current lat point converted to radians
    lon1 = math.radians(lon) # Current long point converted to radians

    lat2 = math.asin(math.sin(lat1)*math.cos(d/R) + math.cos(lat1)*math.sin(d/R)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1), math.cos(d/R)-math.sin(lat1)*math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return (lat2, lon2)

"""
Search Bing for the plane described by the dictionary 'plane'


"""
def image_search(icao24: str, operator: str, type: str, registration: str) -> str:
    """Search Bing for plane images. If found, update planedb with URL

    Arguments:
        icao24 {str} -- ICAO24 designation
        operator {str} -- Operator of aircraft
        type {str} -- Aircraft type
        registration {str} -- Aircraft registration

    Returns:
        str -- URL of image, hopefully

    @todo: don't search for
        Bluebird Nordic Boeing 737 4Q8SF TF-BBM
    but rather
        "Bluebird Nordic" "Boeing 737" "TF-BBM"
    or
        "Bluebird Nordic" "TF-BBM"
    or
        "Bluebird Nordic" Boeing "TF-BBM"
    """
    img_url = None
    # Bing sometimes refuses to search for "Scandinavian Airlines System" :-/
    op = operator.replace("Scandinavian Airlines System", "SAS")
    searchTerm = "%s %s %s" % (op, type, registration)
    logging.debug("Searching for %s", searchTerm)
    imageUrls = bing.imageSearch(searchTerm)
    if not imageUrls:
        imageUrls = bing.imageSearch(registration)
    if imageUrls:
        img_url = imageUrls[0]
        logging.info("Added image %s for %s", img_url, icao24)
#        for k in plane:
#            logging.info("%20s : %s" % (k, plane[k]))
        if not planedb.update_aircraft(icao24, {'image' : img_url}):
            logging.error("Failed to update PlaneDB image for %s" % (icao24))
    else:
        logging.error("Image search came up short for '%s', blacklisted (%s)?" % (searchTerm, icao24))
    return img_url
