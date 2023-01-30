import math
import os

import pandas as pd

import camera
import utils

from matplotlib import pyplot as plt


def point_camera(
        r_XYZ_t,
        E_XYZ_to_ENz,
        e_E_XYZ,
        e_N_XYZ,
        e_z_XYZ,
        alpha,
        beta,
        gamma,
        E_XYZ_to_uvw,
):

    # Convert to specified units of measure
    # currentPlane["lat"]  # [deg]
    # currentPlane["lon"]  # [deg]
    # currentPlane["latLonTime"]
    camera.currentPlane["altitude"] *= 0.3048  # [ft] * [m/ft] = [m]
    # currentPlane["altitudeTime"]
    # currentPlane["track"]  # [deg]
    camera.currentPlane["groundSpeed"] *= (
        6076.12 / 3600 * 0.3048
    )  # [nm/h] * [ft/nm] / [s/h] * [m/ft] = [m/s]
    camera.currentPlane["verticalRate"] *= 0.3048 / 60  # [ft/s] * [m/ft] / [s/m] = [m/s]
    # currentPlane["icao24"]
    # currentPlane["type"]

    r_rst_a_t, v_rst_a_t = camera.calculateCameraPositionB(
        r_XYZ_t,
        E_XYZ_to_ENz,
        e_E_XYZ,
        e_N_XYZ,
        e_z_XYZ,
        alpha,
        beta,
        gamma,
        E_XYZ_to_uvw,
    )

    # distance3dB = camera.distance3d
    # distance2dB = camera.distance2d
    # bearingB = camera.bearing
    # elevationB = camera.elevation
    angularVelocityHorizontalB = camera.angularVelocityHorizontal
    angularVelocityVerticalB = camera.angularVelocityVertical
    cameraPanB = camera.cameraPan
    cameraTiltB = camera.cameraTilt

    return camera.currentPlane["latLonTime"], cameraPanB, cameraTiltB, r_rst_a_t, v_rst_a_t


# Read a processed track file
# data = pd.read_csv("data/A19A08-processed-track.csv")
data = pd.read_csv("data/A1E946-processed-track.csv")

# Assign camerag position
camera.camera_latitude = float(os.getenv("CAMERA_LATITUDE", "38.0"))  # [deg]
camera.camera_longitude = float(os.getenv("CAMERA_LONGITUDE", "-77.0"))  # [deg]
camera.camera_altitude = 86.46  # [m]
camera.camera_lead = 0.0  # [s]

# Assign position of the tripod
t_varphi = camera.camera_latitude  # [deg]
t_lambda = camera.camera_longitude  # [deg]
t_h = camera.camera_altitude  # [m]

# Compute position in the XYZ coordinate system of the tripod
E_XYZ_to_ENz, e_E_XYZ, e_N_XYZ, e_z_XYZ = utils.compute_E(t_lambda, t_varphi)
r_XYZ_t = utils.compute_r_XYZ(t_lambda, t_varphi, t_h)

# Compute the rotations from the XYZ coordinate system to the
# uvw (camera housing fixed) coordinate system
alpha = 0.0  # [deg]
beta = 0.0  # [deg]
gamma = 0.0  # [deg]
q_alpha, q_beta, q_gamma, E_XYZ_to_uvw, _, _, _ = camera.compute_rotations(
    e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, 0.0, 0.0
)

# Initialize the aircraft parameters and compute camera pointing and slew rate
index = 0
camera.currentPlane = data.iloc[index, :].to_dict()
time_a, rho_a, tau_a, r_rst_a_t, v_rst_a_t = point_camera(
    r_XYZ_t,
    E_XYZ_to_ENz,
    e_E_XYZ,
    e_N_XYZ,
    e_z_XYZ,
    alpha,
    beta,
    gamma,
    E_XYZ_to_uvw,
)
dt_a = 0.0
omega = utils.cross(r_rst_a_t, v_rst_a_t) / utils.norm(r_rst_a_t) ** 2
rho_dot_a = math.degrees(-omega[2])
tau_dot_a = math.degrees(omega[0])

# Define control parameters and camera update rate
k_rho = 0.4
k_tau = 1.0
do_init = True
dt_c = 0.1

# Initialize parameters
history = {}
history["time_a"] = [time_a]
history["dt_a"] = [dt_a]
history["rho_a"] = [rho_a]
history["tau_a"] = [tau_a]
history["rho_dot_a"] = [rho_dot_a]
history["tau_dot_a"] = [tau_dot_a]
history["time_c"] = [time_a]
if do_init:
    history["rho_c"] = [rho_a]
    history["tau_c"] = [tau_a]
    history["rho_dot_c"] = [rho_dot_a]
    history["tau_dot_c"] = [tau_dot_a]
else:
    history["rho_c"] = [0]
    history["tau_c"] = [0]
    history["rho_dot_c"] = [0]
    history["tau_dot_c"] = [0]
history["delta_rho_dot_c"] = [0]
history["delta_tau_dot_c"] = [0]

# Step in time at the camera update rate
time_c = history["time_c"][-1]
while index < data.shape[0] - 1:
    time_c += dt_c

    # Assuming constant aircraft velocity, update aircraft position and compute camera slew rate
    r_rst_a_t +=  v_rst_a_t * dt_c
    omega = utils.cross(r_rst_a_t, v_rst_a_t) / utils.norm(r_rst_a_t) ** 2
    delta_rho_dot_c = history["delta_rho_dot_c"][-1]
    delta_tau_dot_c = history["delta_tau_dot_c"][-1]
    rho_dot_c = math.degrees(-omega[2]) + delta_rho_dot_c
    tau_dot_c = math.degrees(omega[0]) + delta_tau_dot_c

    # Update camera pan and tilt
    rho_c = history["rho_c"][-1] + rho_dot_c * dt_c
    tau_c = history["tau_c"][-1] + tau_dot_c * dt_c

    # Update the system inputs at the rate AIS-B messages are recieved
    if time_c >= data["latLonTime"][index + 1]:
        # Update the aircraft parameters and compute camera pointing and slew rate
        index += 1
        camera.currentPlane = data.iloc[index, :].to_dict()
        time_a, rho_a, tau_a, r_rst_a_t, v_rst_a_t = point_camera(
            r_XYZ_t,
            E_XYZ_to_ENz,
            e_E_XYZ,
            e_N_XYZ,
            e_z_XYZ,
            alpha,
            beta,
            gamma,
            E_XYZ_to_uvw,
        )
        omega = utils.cross(r_rst_a_t, v_rst_a_t) / utils.norm(r_rst_a_t) ** 2
        rho_dot_a = math.degrees(-omega[2])
        tau_dot_a = math.degrees(omega[0])

        # Compute slew rate differences
        dt_a = 1.0  # (time_a - history["time_a"][-1])
        delta_rho_dot_c = k_rho * (rho_a - rho_c) / dt_a
        delta_tau_dot_c = k_tau * (tau_a - tau_c) / dt_a
        # delta_rho_dot_c = k_rho * (rho_dot_a - rho_dot_c)
        # delta_tau_dot_c = k_tau * (tau_dot_a - tau_dot_c)

    # Accumulate parameters at each step
    history["time_a"].append(time_a)
    history["dt_a"].append(dt_a)
    history["rho_a"].append(rho_a)
    history["tau_a"].append(tau_a)
    history["rho_dot_a"].append(rho_dot_a)
    history["tau_dot_a"].append(tau_dot_a)
    history["time_c"].append(time_c)
    history["rho_c"].append(rho_c)
    history["tau_c"].append(tau_c)
    history["rho_dot_c"].append(rho_dot_c)
    history["tau_dot_c"].append(tau_dot_c)
    history["delta_rho_dot_c"].append(delta_rho_dot_c)
    history["delta_tau_dot_c"].append(delta_tau_dot_c)

# Convert history dictionary to a data frame
ts = pd.DataFrame.from_dict(history)

# Plot pan angle
fig, axs = plt.subplots(2, 2, figsize=[12.8, 9.6])
axs[0, 0].plot(ts["time_c"], ts["rho_c"] - ts["rho_a"], label="error")
axs[0, 0].plot(ts["time_c"], ts["rho_c"], label="camera")
axs[0, 0].plot(ts["time_c"], ts["rho_a"], label="aircraft")
axs[0, 0].legend()
axs[0, 0].set_title("Camera and Aircraft Pan Angle and Difference")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].set_ylabel("Pan Angle [deg]")

# Plot tilt angle
axs[1, 0].plot(ts["time_c"], ts["tau_c"] - ts["tau_a"], label="error")
axs[1, 0].plot(ts["time_c"], ts["tau_c"], label="camera")
axs[1, 0].plot(ts["time_c"], ts["tau_a"], label="aircraft")
axs[1, 0].legend()
axs[1, 0].set_title("Camera and Aircraft Tilt Angle and Difference")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("Tilt Angle [deg]")

# Plot pan angular rate angle
axs[0, 1].plot(ts["time_c"], ts["rho_dot_c"] - ts["rho_dot_a"], label="error")
axs[0, 1].plot(ts["time_c"], ts["rho_dot_c"], label="camera")
axs[0, 1].plot(ts["time_c"], ts["rho_dot_a"], label="aircraft")
axs[0, 1].legend()
axs[0, 1].set_title("Camera and Aircraft Pan Angular Rate and Difference")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 1].set_ylabel("Pan Anglular Rate [deg/s]")

# Plot tilt angular rate angle
axs[1, 1].plot(ts["time_c"], ts["tau_dot_c"] - ts["tau_dot_a"], label="error")
axs[1, 1].plot(ts["time_c"], ts["tau_dot_c"], label="camera")
axs[1, 1].plot(ts["time_c"], ts["tau_dot_a"], label="aircraft")
axs[1, 1].legend()
axs[1, 1].set_title("Camera and Aircraft Tilt Angular Rate and Difference")
axs[0, 0].set_xlabel("Time [s]")
axs[1, 1].set_ylabel("Tilt Anglular Rate [deg/s]")

plt.show()
