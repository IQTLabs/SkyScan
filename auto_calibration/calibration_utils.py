import logging
import math

import numpy as np


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
