from datetime import datetime, timedelta
import logging
import math
from typing import *

import numpy as np
import quaternion as qn

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

# WGS84 parameters
R_OPLUS = 6378137  # [m]
F_INV = 298.257223563


def deg2rad(deg: float) -> float:
    """Convert degrees to radians

    Arguments:
        deg {float} -- Angle in degrees

    Returns:
        float -- Angle in radians
    """
    return deg * (math.pi / float(180))


def rad2deg(deg: float) -> float:
    """Convert degrees to radians

    Arguments:
        deg {float} -- Angle in degrees

    Returns:
        float -- Angle in radians
    """
    return deg / (math.pi / float(180))


def elevation(distance: float, cameraAltitude, airplaneAltitude):

    if distance > 0:
        ratio = (float(airplaneAltitude) - float(cameraAltitude)) / float(distance)
        a = math.atan(ratio) * (float(180) / math.pi)

        return a
    else:
        logging.info("🚨 Elevation is less than zero 🚨  ")
        return 0


def bearingFromCoordinate(cameraPosition, airplanePosition, heading):
    if heading is None:
        return -1

    lat2 = float(airplanePosition[0])
    lon2 = float(airplanePosition[1])

    lat1 = float(cameraPosition[0])
    lon1 = float(cameraPosition[1])

    dLon = float(lon1 - lon2)

    y = math.sin(dLon * math.pi / float(180)) * math.cos(lat1 * math.pi / float(180))
    x = math.cos(lat2 * math.pi / float(180)) * math.sin(
        lat1 * math.pi / float(180)
    ) - math.sin(lat2 * math.pi / float(180)) * math.cos(
        lat1 * math.pi / float(180)
    ) * math.cos(
        dLon * math.pi / float(180)
    )

    brng = math.atan2(-y, x) * 180 / math.pi
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
    lon1 = cameraPosition[1]

    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    rlon1 = math.radians(lon1)
    rlon2 = math.radians(lon2)
    dlon = math.radians(lon2 - lon1)

    b = math.atan2(
        math.sin(dlon) * math.cos(rlat2),
        math.cos(rlat1) * math.sin(rlat2)
        - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon),
    )  # bearing calc
    bd = math.degrees(b)
    br, bn = divmod(bd + 360, 360)  # the bearing remainder and final bearing

    return bn


def coordinate_distance_3d(
    lat1: float, lon1: float, alt1: float, lat2: float, lon2: float, alt2: float
) -> float:
    """Calculate distance in meters between the two coordinates

    Arguments:
        lat1 {float} -- Start latitude (deg)
        lon1 {float} -- Start longitude (deg)
        alt1 {float} -- Start altitude (meters)
        lat2 {float} -- End latitude (deg)
        lon2 {float} -- End longitude (deg)
        alt2 {float} -- End altitude (meters)

    Returns:
        float -- Distance in meters
    """
    R = 6371 # Radius of the earth in km
    dLat = deg2rad(lat2 - lat1)
    dLon = deg2rad(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(deg2rad(lat1)) * math.cos(
        deg2rad(lat2)
    ) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c * 1000 #  Distance in m
    # return d
    # logging.info("Alt1: " + str(alt1) + " Alt2: " + str(alt2))
    alt_diff = abs(alt1 - alt2)

    rtt = (d**2 + alt_diff**2) ** 0.5

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
    dLat = deg2rad(lat2 - lat1)
    dLon = deg2rad(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(deg2rad(lat1)) * math.cos(
        deg2rad(lat2)
    ) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c * 1000 #  Distance in m
    return d


def calc_travel(
    lat: float,
    lon: float,
    utc_start: datetime,
    speed_mps: float,
    heading: float,
    lead_s: float,
) -> Tuple[float, float]:
    """Calculate travel from lat, lon starting at a certain time with given speed and heading

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

    R = 6371  # Radius of the Earth in km
    brng = math.radians(heading)  # Bearing is 90 degrees converted to radians.
    d = (age_s * speed_mps) / 1000.0  # Distance in km

    lat1 = math.radians(lat)  # Current lat point converted to radians
    lon1 = math.radians(lon)  # Current long point converted to radians

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d / R)
        + math.cos(lat1) * math.sin(d / R) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d / R) * math.cos(lat1),
        math.cos(d / R) - math.sin(lat1) * math.sin(lat2),
    )

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return (lat2, lon2)


def convert_time(inp_date_time):
    try:
        out_date_time = datetime.strptime(inp_date_time, "%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        logger.warning(
            f"Could not parse latLonTime as string with decimal seconds: {e}"
        )

    try:
        out_date_time = datetime.strptime(inp_date_time, "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.warning(f"Could not parse latLonTime as string: {e}")

    try:
        out_date_time = datetime.fromtimestamp(inp_date_time)
    except Exception as e:
        logger.warning(f"Could not construct datetime for latLonTime as timestamp: {e}")

    return out_date_time


def calc_travel_3d(current_plane, lead_s: float):
    """Extrapolate the 3D position of the aircraft

    Arguments:
        lat {float} -- Starting latitude (degrees)
        lon {float} -- Starting longitude (degrees)
        alt {float} -- Starting altitude (meters)
        lat_lon_time {datetime} -- Last time lat / lon was updated
        altitude_time {datetime} -- Last time altitude was updated
        speed_mps {float} -- Speed (meters per second)
        heading {float} -- Heading (degrees)
        climb_rate {float} -- climb rate (meters per second)

    Returns:
        Tuple[float, float, float] -- The new latitude (deg)/longitude (deg)/alt (meters) as a tuple
    """
    lat = current_plane["lat"]
    lon = current_plane["lon"]
    alt = current_plane["altitude"]

    lat_lon_time = convert_time(current_plane["latLonTime"])
    altitude_time = convert_time(current_plane["altitudeTime"])

    speed_mps = current_plane["groundSpeed"]
    heading = current_plane["track"]
    climb_rate = current_plane["verticalRate"]

    # TODO: Restore
    # lat_lon_age = datetime.utcnow() - lat_lon_time
    # lat_lon_age_s = lat_lon_age.total_seconds() + lead_s
    lat_lon_age_s = lead_s

    # TODO: Restore
    # alt_age = datetime.utcnow() - altitude_time
    # alt_age_s = alt_age.total_seconds() + lead_s
    alt_age_s = lead_s

    R = float(6371) # Radius of the Earth in km
    brng = math.radians(heading)  # Bearing is 90 degrees converted to radians.
    d = float((lat_lon_age_s * speed_mps) / 1000.0) # Distance in km

    lat1 = math.radians(lat)  # Current lat point converted to radians
    lon1 = math.radians(lon)  # Current long point converted to radians

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d / R)
        + math.cos(lat1) * math.sin(d / R) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d / R) * math.cos(lat1),
        math.cos(d / R) - math.sin(lat1) * math.sin(lat2),
    )

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    alt2 = alt + climb_rate * alt_age_s

    return (lat2, lon2, alt2)


def angular_velocity(currentPlane, camera_latitude, camera_longitude, camera_altitude):
    (lat, lon, alt) = calc_travel_3d(currentPlane, 0)
    distance2d = coordinate_distance(camera_latitude, camera_longitude, lat, lon)
    bearing1 = bearingFromCoordinate(
        cameraPosition=[camera_latitude, camera_longitude],
        airplanePosition=[lat, lon],
        heading=currentPlane["track"],
    )
    elevation1 = elevation(
        distance2d, cameraAltitude=camera_altitude, airplaneAltitude=alt
    )

    (lat, lon, alt) = calc_travel_3d(currentPlane, 1)
    distance2d = coordinate_distance(camera_latitude, camera_longitude, lat, lon)
    bearing2 = bearingFromCoordinate(
        cameraPosition=[camera_latitude, camera_longitude],
        airplanePosition=[lat, lon],
        heading=currentPlane["track"],
    )
    elevation2 = elevation(
        distance2d, cameraAltitude=camera_altitude, airplaneAltitude=alt
    )

    angularVelocityH = ((bearing2 - bearing1) + 180) % 360 - 180
    angularVelocityV = elevation2 - elevation1

    return (angularVelocityH, angularVelocityV)


def compute_e_E_XYZ(d_lambda):
    """Compute components of the east unit vector at a given geodetic
    longitude and latitude.

    Parameters
    ----------
    d_lambda : float
        Geodetic longitude [deg]

    Returns
    -------
    e_E_XYZ : numpy.ndarray
        Components of the east unit vector in an Earth
        fixed geocentric equatorial coordinate system
    """
    r_lambda = math.radians(d_lambda)
    e_E_XYZ = np.array([-math.sin(r_lambda), math.cos(r_lambda), 0])
    return e_E_XYZ


def compute_e_N_XYZ(d_lambda, d_varphi):
    """Compute components of the north unit vector at a
    given geodetic longitude and latitude.

    Parameters
    ----------
    d_lambda : float
        Geodetic longitude [deg]
    d_varphi : float
        Geodetic latitude [deg]

    Returns
    -------
    e_N_XYZ : numpy.ndarray
        Components of the north unit vector in an Earth
        fixed geocentric equatorial coordinate system
    """
    r_lambda = math.radians(d_lambda)
    r_varphi = math.radians(d_varphi)
    e_N_XYZ = np.array(
        [
            -math.sin(r_varphi) * math.cos(r_lambda),
            -math.sin(r_varphi) * math.sin(r_lambda),
            math.cos(r_varphi),
        ]
    )
    return e_N_XYZ


def compute_e_z_XYZ(d_lambda, d_varphi):
    """Compute components of the zenith unit vector at a
    given geodetic longitude and latitude.

    Parameters
    ----------
    d_lambda : float
        Geodetic longitude [deg]
    d_varphi : float
        Geodetic latitude [deg]

    Returns
    -------
    e_z_XYZ : numpy.ndarray
        Components of the zenith unit vector in an Earth
        fixed geocentric equatorial coordinate system
    """
    r_lambda = math.radians(d_lambda)
    r_varphi = math.radians(d_varphi)
    e_z_XYZ = np.array(
        [
            math.cos(r_varphi) * math.cos(r_lambda),
            math.cos(r_varphi) * math.sin(r_lambda),
            math.sin(r_varphi),
        ]
    )
    return e_z_XYZ


def compute_E(d_lambda, d_varphi):
    """Compute orthogonal transformation matrix from geocentric to
    topocentric coordinates.

    Parameters
    ----------
    d_lambda : float
        Geodetic longitude [deg]
    d_varphi : float
        Geodetic latitude [deg]

    Returns
    -------
    E : numpy.ndarray
        Orthogonal transformation matrix from geocentric to
        topocentric coordinates
    """
    e_E_XYZ = compute_e_E_XYZ(d_lambda)
    e_N_XYZ = compute_e_N_XYZ(d_lambda, d_varphi)
    e_z_XYZ = compute_e_z_XYZ(d_lambda, d_varphi)
    E_XYZ_to_ENz = np.row_stack((e_E_XYZ, e_N_XYZ, e_z_XYZ))
    return E_XYZ_to_ENz, e_E_XYZ, e_N_XYZ, e_z_XYZ


def compute_r_XYZ(d_lambda, d_varphi, o_h):
    """Compute the position given geodetic longitude and
    latitude, and altitude.

    Parameters
    ----------
    d_lambda : float or numpy.ndarray
        Geodetic longitude [deg]
    d_varphi : float or numpy.ndarray
        Geodetic latitude [deg]
    o_h : float or numpy.ndarray
        Altitude [m]

    Returns
    -------
    r_XYZ : numpy.ndarray
        Position [m] in an Earth
        fixed geocentric equatorial coordinate system
    """
    f = 1 / F_INV
    if type(d_lambda) == float:
        r_lambda = math.radians(d_lambda)
        r_varphi = math.radians(d_varphi)
        N = R_OPLUS / math.sqrt(1 - f * (2 - f) * math.sin(r_varphi) ** 2)
        r_XYZ = np.array(
            [
                (N + o_h) * math.cos(r_varphi) * math.cos(r_lambda),
                (N + o_h) * math.cos(r_varphi) * math.sin(r_lambda),
                ((1 - f) ** 2 * N + o_h) * math.sin(r_varphi),
            ]
        )
    elif type(d_lambda) == np.ndarray:
        r_lambda = np.radians(d_lambda)
        r_varphi = np.radians(d_varphi)
        N = R_OPLUS / np.sqrt(1 - f * (2 - f) * np.sin(r_varphi) ** 2)
        r_XYZ = np.row_stack(
            (
                (N + o_h) * np.cos(r_varphi) * np.cos(r_lambda),
                (N + o_h) * np.cos(r_varphi) * np.sin(r_lambda),
                ((1 - f) ** 2 * N + o_h) * np.sin(r_varphi),
            ),
        )
    return r_XYZ


def as_quaternion(s, v):
    """Construct a quaternion given a scalar and vector.

    Parameters
    ----------
    s : float
        A scalar value
    v : list or numpy.ndarray
        A vector of floats

    Returns
    -------
    quaternion.quaternion
        A quaternion with the specified scalar and vector parts
    """
    if type(s) != float:
        raise Exception("Scalar part is not a float")
    if len(v) != 3 or not all([type(e) == float or type(e) == np.float64 for e in v]):
        raise Exception("Vector part is not an iterable of three floats")
    return np.quaternion(s, v[0], v[1], v[2])


def as_rotation_quaternion(d_omega, u):
    """Construct a rotation quaternion given an angle and direction of
    rotation.

    Parameters
    ----------
    d_omega : float
        An angle [deg]
    u : list or numpy.ndarray
        A vector of floats

    Returns
    -------
    quaternion.quaternion
        A rotation quaternion with the specified angle and direction
    """
    if type(d_omega) != float:
        raise Exception("Angle is not a float")
    if len(u) != 3 or not all([type(e) == float or type(e) == np.float64 for e in u]):
        raise Exception("Vector part is not an iterable of three floats")
    r_omega = math.radians(d_omega)
    v = [math.sin(r_omega / 2) * e for e in u]
    return np.quaternion(math.cos(r_omega / 2), v[0], v[1], v[2])


def as_vector(q):
    """Return the vector part of a quaternion, provided the scalar
    part is nearly zero.

    Parameters
    ----------
    q : quaternion.quaternion
        A vector quaternion

    Returns
    -------
    numpy.ndarray
       A vector of floats
    """
    if math.fabs(q.w) > 1e-12:
        raise Exception("Quaternion is not a vector quaternion")
    return np.array([q.x, q.y, q.z])


def compute_great_circle_distance(varphi_1, lambda_1, varphi_2, lambda_2):
    """Use the haversine formula to compute the great-circle distance
    between two points on a sphere given their longitudes and
    latitudes.

    See:
        https://en.wikipedia.org/wiki/Haversine_formula

    Parameters
    ----------
    varphi_1 : float
        Latitude [deg]
    lambda_1 : float
        Longitude [deg]
    varphi_2 : float
        Latitude [deg]
    lambda_2 : float
        Longitude [deg]

    Returns
    -------
    float
        Great-circle distance [m]

    """
    return (
        2
        * R_OPLUS
        * math.asin(
            math.sqrt(
                math.sin(math.radians((varphi_2 - varphi_1) / 2.0)) ** 2
                + math.cos(math.radians(varphi_1))
                * math.cos(math.radians(varphi_2))
                * math.sin(math.radians((lambda_2 - lambda_1) / 2.0)) ** 2
            )
        )
    )
