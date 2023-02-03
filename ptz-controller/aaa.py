import math
import os

import numpy as np
import pandas as pd

import ptz_utilities


# Read a processed track file
# data = pd.read_csv("../axis-ptz/data/A19A08-processed-track.csv")
data = pd.read_csv("../axis-ptz/data/A1E946-processed-track.csv")


index = 0
payload = data.iloc[index, :].to_dict()
# Convert to specified units of measure
# payload["lat"]  # [deg]
# payload["lon"]  # [deg]
# payload["latLonTime"]
payload["altitude"] *= 0.3048  # [ft] * [m/ft] = [m]
# payload["altitudeTime"]
# payload["track"]  # [deg]
payload["groundSpeed"] *= (
    6076.12 / 3600 * 0.3048
)  # [nm/h] * [ft/nm] / [s/h] * [m/ft] = [m/s]
payload["verticalRate"] *= 0.3048 / 60  # [ft/s] * [m/ft] / [s/m] = [m/s]
# payload["icao24"]
# payload["type"]


lambda_t = float(os.getenv("TRIPOD_LONGITUDE"))
varphi_t = float(os.getenv("TRIPOD_LATITUDE"))
h_t = 0.0

# Compute tripod position in the geocentric (XYZ) coordinate
# system
r_XYZ_t = ptz_utilities.compute_r_XYZ(lambda_t, varphi_t, h_t)

# Compute orthogonal transformation matrix from geocentric
# (XYZ) to topocentric (ENz) coordinates
E_XYZ_to_ENz, e_E_XYZ, e_N_XYZ, e_z_XYZ = ptz_utilities.compute_E_XYZ_to_ENz(
    lambda_t, varphi_t
)


# Compute the rotations from the geocentric (XYZ) coordinate
# system to the camera housing fixed (uvw) coordinate system

# Case A
alpha = 0.0
beta = 0.0
gamma = 0.0
rho = 0.0
tau = 0.0
(
    q_alpha,
    q_beta,
    q_gamma,
    E_XYZ_to_uvw_A,
    _,
    _,
    _,
) = ptz_utilities.compute_camera_rotations(
    e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, rho, tau
)

# Case b
alpha = 0.0
beta = 45.0
gamma = 0.0
rho = 0.0
tau = 0.0
(
    q_alpha,
    q_beta,
    q_gamma,
    E_XYZ_to_uvw_B,
    _,
    _,
    _,
) = ptz_utilities.compute_camera_rotations(
    e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, rho, tau
)


# Assign position and velocity of the aircraft
time_a = payload["latLonTime"]  # [sec]
lambda_a = payload["lon"]  # [deg]
varphi_a = payload["lat"]  # [deg]
h_a = payload["altitude"]  # [m]
track_a = payload["track"]  # [deg]
ground_speed_a = payload["groundSpeed"]  # [m/s]
vertical_rate_a = payload["verticalRate"]  # [m/s]

# Compute position in the geocentric (XYZ) coordinate system
# of the aircraft relative to the tripod at time zero, the
# observation time
r_XYZ_a_0 = ptz_utilities.compute_r_XYZ(lambda_a, varphi_a, h_a)
r_XYZ_a_0_t = r_XYZ_a_0 - r_XYZ_t

# Compute position and velocity in the topocentric (ENz)
# coordinate system of the aircraft relative to the tripod at
# time zero, and position at slightly later time one
r_ENz_a_0_t = np.matmul(E_XYZ_to_ENz, r_XYZ_a_0_t)
track_a = math.radians(track_a)
v_ENz_a_0_t = np.array(
    [
        ground_speed_a * math.sin(track_a),
        ground_speed_a * math.cos(track_a),
        vertical_rate_a,
    ]
)
lead_time = 0.25
r_ENz_a_1_t = r_ENz_a_0_t + v_ENz_a_0_t * lead_time

# Compute position, at time one, and velocity, at time zero,
# in the geocentric (XYZ) coordinate system of the aircraft
# relative to the tripod
r_XYZ_a_1_t = np.matmul(E_XYZ_to_ENz.transpose(), r_ENz_a_1_t)
v_XYZ_a_0_t = np.matmul(E_XYZ_to_ENz.transpose(), v_ENz_a_0_t)

# Compute pan and tilt to point the camera at the aircraft

# Case A
r_uvw_a_1_t = np.matmul(E_XYZ_to_uvw_A, r_XYZ_a_1_t)
rho_A = math.degrees(math.atan2(r_uvw_a_1_t[0], r_uvw_a_1_t[1]))  # [deg]
tau_A = math.degrees(
    math.atan2(r_uvw_a_1_t[2], ptz_utilities.norm(r_uvw_a_1_t[0:2]))
)  # [deg]


# Case B
r_uvw_a_1_t = np.matmul(E_XYZ_to_uvw_B, r_XYZ_a_1_t)
rho_B = math.degrees(math.atan2(r_uvw_a_1_t[0], r_uvw_a_1_t[1]))  # [deg]
tau_B = math.degrees(
    math.atan2(r_uvw_a_1_t[2], ptz_utilities.norm(r_uvw_a_1_t[0:2]))
)  # [deg]
