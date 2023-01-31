import logging
import math

import numpy as np
import quaternion

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

# WGS84 parameters
R_OPLUS = 6378137  # [m]
F_INV = 298.257223563


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


def compute_E_XYZ_to_ENz(d_lambda, d_varphi):
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
    E_XYZ_to_ENz : numpy.ndarray
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
    v : numpy.ndarray
        A vector of floats

    Returns
    -------
    quaternion.quaternion
        A quaternion with the specified scalar and vector parts
    """
    return np.quaternion(s, v[0], v[1], v[2])


def as_rotation_quaternion(d_omega, u):
    """Construct a rotation quaternion given an angle and direction of
    rotation.

    Parameters
    ----------
    d_omega : float
        An angle [deg]
    u : numpy.ndarray
        A vector of floats

    Returns
    -------
    quaternion.quaternion
        A rotation quaternion with the specified angle and direction
    """
    r_omega = math.radians(d_omega)
    v = math.sin(r_omega / 2.0) * u
    return np.quaternion(math.cos(r_omega / 2.0), v[0], v[1], v[2])


def as_vector(q):
    """Return the vector part of a quaternion.

    Parameters
    ----------
    q : quaternion.quaternion
        A quaternion, assumed to be a vector quaternion with scalar part zero

    Returns
    -------
    numpy.ndarray
       A vector of floats
    """
    return np.array([q.x, q.y, q.z])


def cross(u, v):
    """Compute the cross product of two vectors.

    Parameters
    ----------
    u: numpy.ndarray
       A vector of floats
    v: numpy.ndarray
       A vector of floats

    Returns
    -------
    v: numpy.ndarray
       The cross product vector of floats
    """
    w = np.array([0.0, 0.0, 0.0])
    w[0] = u[1] * v[2] - u[2] * v[1]
    w[1] = u[2] * v[0] - u[0] * v[2]
    w[2] = u[0] * v[1] - u[1] * v[0]
    return w


def norm(v):
    """Compute the Euclidean norm of a vector.

    Parameters
    ----------
    v: numpy.ndarray
       A vector of floats

    Returns
    -------
    float
       the Euclidean norm of the vector
    """
    s = 0
    for i in range(len(v)):
        s += v[i]**2
    return math.sqrt(s)


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
