from argparse import ArgumentParser
import json
import logging
import os
import time

from matplotlib import pyplot as plt
import pandas as pd
import schedule

import ptz_controller

HEARTBEAT_INTERVAL = 10.0
UPDATE_INTERVAL = 0.01
CAPTURE_INTERVAL = 2.0
LEAD_TIME = 0.0
PAN_GAIN = 0.1
PAN_RATE_MIN = 1.0
PAN_RATE_MAX = 100.0
TILT_GAIN = 0.4
TILT_RATE_MIN = 1.0
TILT_RATE_MAX = 100.0
JPEG_RESOLUTION = "1920x1080"
JPEG_COMPRESSION = 5

logger = logging.getLogger("ptz-integration")
logger.setLevel(logging.INFO)


def make_controller(use_mqtt):
    """Construct a controller.

    Note that if use_mqtt = True then an MQTT broker must be started
    manually.

    Parameters
    ----------
    use_mqtt: bool
        Flag to use MQTT, or not

    Returns
    -------
    None
    """
    controller = ptz_controller.PtzController(
        camera_ip=os.getenv("CAMERA_IP"),
        camera_user=os.getenv("CAMERA_USER"),
        camera_password=os.getenv("CAMERA_PASSWORD"),
        mqtt_ip=os.getenv("MQTT_IP"),
        config_topic=os.getenv("CONFIG_TOPIC"),
        calibration_topic=os.getenv("CALIBRATION_TOPIC"),
        flight_topic=os.getenv("FLIGHT_TOPIC"),
        capture_topic=os.getenv("CAPTURE_TOPIC"),
        logger_topic=os.getenv("LOGGER_TOPIC"),
        heartbeat_interval=HEARTBEAT_INTERVAL,
        update_interval=UPDATE_INTERVAL,
        capture_interval=CAPTURE_INTERVAL,
        lead_time=LEAD_TIME,
        pan_gain=PAN_GAIN,
        pan_rate_min=PAN_RATE_MIN,
        pan_rate_max=PAN_RATE_MAX,
        tilt_gain=TILT_GAIN,
        tilt_rate_min=TILT_RATE_MIN,
        tilt_rate_max=TILT_RATE_MAX,
        jpeg_resolution=JPEG_RESOLUTION,
        jpeg_compression=JPEG_COMPRESSION,
        use_mqtt=use_mqtt,
        use_camera=False,
        include_age=False,
    )
    return controller


def get_config_msg():
    """Populate a config message, reading actual, by private longitude
    and latitude from the environment.

    Parameters
    ----------
    None

    Returns
    -------
    msg : dict
        The configuration message
    """
    msg = {}
    msg["data"] = {}
    msg["data"]["tripod_longitude"] = float(
        os.getenv("TRIPOD_LONGITUDE", "-77.0")
    )  # [deg]
    msg["data"]["tripod_latitude"] = float(
        os.getenv("TRIPOD_LATITUDE", "38.0")
    )  # [deg]
    msg["data"]["tripod_altitude"] = 86.46  # [m]
    return msg


def get_calibration_msg():
    """Populate a calibration message with all 0 deg angles.

    Parameters
    ----------
    None

    Returns
    -------
    msg : dict
        The calibration message
    """
    with open("data/calibration_msg_0s.json", "r") as f:
        msg = json.load(f)
    return msg


def make_flight_msg(track, index):
    """Populate a flight message with track data at the specified
    index.

    Parameters
    ----------
    track : pd.DataFrame()
        The track data
    index : int
        The index of the track data to use

    Returns
    -------
    msg : dict
        The flight message
    """
    msg = {}
    msg["data"] = track.iloc[index, :].to_dict()
    return msg


def read_track_data(track_id):
    """Read a track file and convert to standard units of measure.

    Parameters
    ----------
    track_id : str
        The track identifier

    Returns
    -------
    track : pd.DataFrame()
        The track data read from the file
    """
    track = pd.read_csv(f"data/{track_id}-processed-track.csv")
    track["altitude"] *= 0.3048  # [ft] * [m/ft] = [m]
    track["groundSpeed"] *= (
        6076.12 / 3600 * 0.3048
    )  # [nm/h] * [ft/nm] / [s/h] * [m/ft] = [m/s]
    track["verticalRate"] *= 0.3048 / 60  # [ft/s] * [m/ft] / [s/m] = [m/s]
    return track


def plot_time_series(ts):
    """Plot time series produced by processing messages.

    Parameters
    ----------
    ts : pd.DataFrame()
        Dataframe containing time series to plot

    Returns
    -------
    None
    """

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


def main():
    """Read a track file and process the corresponding messages using
    MQTT, or not.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Provide for some command line arguments
    parser = ArgumentParser(
        description="Read a track file and process the corresponding messages"
    )
    parser.add_argument(
        "-t",
        "--track-id",
        default="A1E946",
        help="The track identifier to process: A1E946 (the default) or A19A08",
    )
    parser.add_argument(
        "-m",
        "--use-mqtt",
        action="store_true",
        help="Use MQTT to process messages",
    )
    args = parser.parse_args()

    # Read the track data
    logger.info(f"Reading track for id: {args.track_id}")
    track = read_track_data(args.track_id)

    # Make the controller, subscribe to all topics, and publish, or
    # process, one message to each topic
    logger.info("Making the controller, and subscribing to topics")
    controller = make_controller(args.use_mqtt)
    controller.add_subscribe_topic(controller.config_topic, controller._config_callback)
    controller.add_subscribe_topic(
        controller.calibration_topic, controller._calibration_callback
    )
    controller.add_subscribe_topic(controller.flight_topic, controller._flight_callback)
    config_msg = get_config_msg()
    calibration_msg = get_calibration_msg()
    index = 0
    flight_msg = make_flight_msg(track, index)
    if controller.use_mqtt:
        logger.info(f"Publishing config msg: {config_msg}")
        controller.publish_to_topic(controller.config_topic, json.dumps(config_msg))
        time.sleep(UPDATE_INTERVAL)
        logger.info(f"Publishing calibration msg: {calibration_msg}")
        controller.publish_to_topic(
            controller.calibration_topic, json.dumps(calibration_msg)
        )
        time.sleep(UPDATE_INTERVAL)
        logger.info(f"Publishing flight msg: {flight_msg}")
        controller.publish_to_topic(controller.flight_topic, json.dumps(flight_msg))
        time.sleep(UPDATE_INTERVAL)

    else:
        _client = None
        _userdata = None
        controller._config_callback(_client, _userdata, config_msg)
        controller._calibration_callback(_client, _userdata, calibration_msg)
        controller._flight_callback(_client, _userdata, flight_msg)

    # Initialize history for plotting
    history = {}
    history["time_c"] = [flight_msg["data"]["latLonTime"]]
    history["rho_a"] = [controller.rho_a]
    history["tau_a"] = [controller.tau_a]
    history["rho_dot_a"] = [controller.rho_dot_a]
    history["tau_dot_a"] = [controller.tau_dot_a]
    history["rho_c"] = [controller.rho_c]
    history["tau_c"] = [controller.tau_c]
    history["rho_dot_c"] = [controller.rho_dot_c]
    history["tau_dot_c"] = [controller.tau_dot_c]

    # Loop in camera time
    dt_c = controller.update_interval
    time_c = history["time_c"][0]
    while index < track.shape[0] - 1:
        time_c += dt_c

        # Process each flight message when received
        if time_c >= track["latLonTime"][index + 1]:
            index = track["latLonTime"][time_c >= track["latLonTime"]].index[-1]
            flight_msg = make_flight_msg(track, index)
            logger.info(f"Publishing flight msg: {flight_msg}")
            if controller.use_mqtt:
                controller.publish_to_topic(
                    controller.flight_topic, json.dumps(flight_msg)
                )
                time.sleep(UPDATE_INTERVAL)

            else:
                controller._flight_callback(_client, _userdata, flight_msg)

        # Always update pointing
        controller._update_pointing()

        # Append to history for plotting
        history["time_c"].append(time_c)
        history["rho_a"].append(controller.rho_a)
        history["tau_a"].append(controller.tau_a)
        history["rho_dot_a"].append(controller.rho_dot_a)
        history["tau_dot_a"].append(controller.tau_dot_a)
        history["rho_c"].append(controller.rho_c)
        history["tau_c"].append(controller.tau_c)
        history["rho_dot_c"].append(controller.rho_dot_c)
        history["tau_dot_c"].append(controller.tau_dot_c)

    # Convert history dictionary to a dataframe, and plot
    ts = pd.DataFrame.from_dict(history)
    plot_time_series(ts)


if __name__ == "__main__":
    main()
