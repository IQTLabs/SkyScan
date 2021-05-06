from typing import *
import logging
import math
from datetime import datetime, timedelta


def deg2rad(deg: float) -> float:
    """Convert degrees to radians

    Arguments:
        deg {float} -- Angle in degrees

    Returns:
        float -- Angle in radians
    """
    return deg * (math.pi/180)

def elevation(distance: float, cameraAltitude, airplaneAltitude):

    if distance > 0:
        ratio = ( airplaneAltitude - cameraAltitude) / float(distance)
        a = math.atan(ratio) * (180 /math.pi)
        return a
    else:
        return 0

def bearingFromCoordinate( cameraPosition, airplanePosition, heading):
    if heading is None:
        return -1
        
    lat2 = airplanePosition[0]
    lon2 = airplanePosition[1]
    
    lat1 = cameraPosition[0]
    lon1 = cameraPosition[1]
    
    dLon = (lon1 - lon2)

    y = math.sin(dLon*math.pi/180) * math.cos(lat1*math.pi/180)
    x = math.cos(lat2*math.pi/180) * math.sin(lat1*math.pi/180) - math.sin(lat2*math.pi/180) * math.cos(lat1*math.pi/180) * math.cos(dLon*math.pi/180)

    brng = math.atan2(-y, x)*180/math.pi
    brng = (brng + 360) % 360
    brng = 360 - brng
    brng -= heading
    brng = (brng + 360) % 360
    return brng


def cameraPanFromCoordinate(airplanePosition, cameraPosition) -> float:
    """Calculate bearing from lat1/lon2 to lat2/lon2

    Arguments:
        lat1 {float} -- Start latitude
        lon1 {float} -- Start longitude
        lat2 {float} -- End latitude
        lon2 {float} -- End longitude

    Returns:
        float -- bearing in degrees
    """

    lat2 = airplanePosition[0]
    lon2 = airplanePosition[1]
    
    lat1 = cameraPosition[0]
    long1 = cameraPosition[1]
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    rlon1 = math.radians(lon1)
    rlon2 = math.radians(lon2)
    dlon = math.radians(lon2-lon1)

    b = math.atan2(math.sin(dlon)*math.cos(rlat2),math.cos(rlat1)*math.sin(rlat2)-math.sin(rlat1)*math.cos(rlat2)*math.cos(dlon)) # bearing calc
    bd = math.degrees(b)
    br,bn = divmod(bd+360,360) # the bearing remainder and final bearing

    return bn

def coordinate_distance_3d(lat1: float, lon1: float, alt1: float, lat2: float, lon2: float, alt2: float) -> float:
    """Calculate distance in meters between the two coordinates

    Arguments:
        lat1 {float} -- Start latitude
        lon1 {float} -- Start longitude
        alt1 {float} -- Start altitude
        lat2 {float} -- End latitude
        lon2 {float} -- End longitude
        alt2 {float} -- End altitude

    Returns:
        float -- Distance in meters
    """
    R = 6371 # Radius of the earth in km
    dLat = deg2rad(lat2-lat1)
    dLon = deg2rad(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c * 1000 #  Distance in m
    #return d
    #logging.info("Alt1: " + str(alt1) + " Alt2: " + str(alt2))
    alt_diff = abs(alt1 - alt2)
    
    rtt = ((d**2 + alt_diff**2)**0.5)

    return rtt

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


def calc_travel(lat: float, lon: float, utc_start: datetime, speed_kts: float, heading: float, lead_s: float) -> Tuple[float, float]:
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
    age_s = age.total_seconds() + lead_s

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

