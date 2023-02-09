from argparse import ArgumentParser
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict

import paho.mqtt.client as mqtt
import pandas as pd

# TODO: Agree on a method for importing the base class
sys.path.append(str(Path(os.getenv("CORE_PATH")).expanduser()))
from base_mqtt_pub_sub import BaseMQTTPubSub
from test_integration import (
    read_track_data,
    get_config_msg,
    get_calibration_msg,
    make_flight_msg,
    plot_time_series,
)

UPDATE_INTERVAL = 0.1

logger = logging.getLogger("ptz-messages")
logger.setLevel(logging.INFO)


class MessageHandler(BaseMQTTPubSub):
    """Subscribe to all required topics, and open files for logging."""

    def __init__(
        self: Any,
        config_topic: str,
        calibration_topic: str,
        flight_topic: str,
        logger_topic: str,
        **kwargs: Any,
    ):
        """Initialize a MessageHandler by subscribing to all required
        topics, connecting to the MQTT broker, and opening logging
        files.

        Parameters
        ----------
        config_topic: str
            MQTT topic for publishing or subscribing to configuration
            messages
        calibration_topic: str
            MQTT topic for publishing or subscribing to calibration
            messages
        flight_topic: str
            MQTT topic for publishing or subscribing to flight
            messages
        logger_topic: str
            MQTT topic for publishing or subscribing to logger
            messages

        Returns
        -------
        MessageHandler
        """
        # Parent class handles kwargs, including MQTT IP
        super().__init__(**kwargs)
        self.config_topic = config_topic
        self.calibration_topic = calibration_topic
        self.flight_topic = flight_topic
        self.logger_topic = logger_topic

        # Connect MQTT client
        logger.info("Connecting MQTT client")
        self.connect_client()
        time.sleep(1)
        self.publish_registration("Message Handler Module Registration")

        # Open files for logging
        self.camera_pointing_filename = "camera-pointing.csv"
        self.camera_pointing_file = open(self.camera_pointing_filename, "w")
        self.camera_pointing_keys = [
            "time_c",
            "rho_a",
            "tau_a",
            "rho_dot_a",
            "tau_dot_a",
            "rho_c",
            "tau_c",
            "rho_dot_c",
            "tau_dot_c",
        ]
        self.camera_pointing_file.write(",".join(self.camera_pointing_keys) + "\n")

    def _logger_callback(
        self: Any, _client: mqtt.Client, _userdata: Dict[Any, Any], msg: Any
    ) -> None:
        """
        Process logging message based on keys.

        Parameters
        ----------
        _client: mqtt.Client
            MQTT client
        _userdata: dict
            Any required user data
        msg: Any
            A JSON string with {timestamp: ____, data: ____,}

        Returns
        -------
        None
        """
        data = self.decode_payload(msg)
        if "camera-pointing" in data:
            p = data["camera-pointing"]
            self.camera_pointing_file.write(
                ",".join([str(p[k]) for k in self.camera_pointing_keys]) + "\n"
            )
        elif "info" in data:
            logger.info(data["info"]["message"])


def make_handler():
    """Construct a MessageHandler.

    Note that an MQTT broker must be started manually.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    handler = MessageHandler(
        mqtt_ip=os.environ.get("MQTT_IP"),
        config_topic=os.environ.get("CONFIG_TOPIC"),
        calibration_topic=os.environ.get("CALIBRATION_TOPIC"),
        flight_topic=os.environ.get("FLIGHT_TOPIC"),
        logger_topic=os.environ.get("LOGGER_TOPIC"),
    )
    return handler


def main():
    """Read a track file and publish the corresponding messages using
    MQTT. Subscribe to the logger topic and process log messages to
    collect camera pointing time series for plotting.

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
    args = parser.parse_args()

    # Read the track data
    logger.info(f"Reading track for id: {args.track_id}")
    track = read_track_data(args.track_id)

    # Make the controller, and subscribe to the logger topic
    logger.info("Making the handler, and subscribing to topics")
    handler = make_handler()
    handler.add_subscribe_topic(handler.logger_topic, handler._logger_callback)
    logger_msg = {
        "timestamp": str(int(datetime.utcnow().timestamp())),
        "data": {
            "info": {
                "message": "Subscribed to the logger",
            }
        },
    }
    handler.publish_to_topic(handler.logger_topic, json.dumps(logger_msg))

    # Publish the configuration and calibration message, and the first
    # flight message
    config_msg = get_config_msg()
    calibration_msg = get_calibration_msg()
    index = 0
    flight_msg = make_flight_msg(track, index)
    logger.info(f"Publishing config msg: {config_msg}")
    handler.publish_to_topic(handler.config_topic, json.dumps(config_msg))
    time.sleep(UPDATE_INTERVAL)
    logger.info(f"Publishing calibration msg: {calibration_msg}")
    handler.publish_to_topic(handler.calibration_topic, json.dumps(calibration_msg))
    time.sleep(UPDATE_INTERVAL)
    logger.info(f"Publishing flight msg: {flight_msg}")
    handler.publish_to_topic(handler.flight_topic, json.dumps(flight_msg))
    time.sleep(UPDATE_INTERVAL)

    # Loop in camera time
    dt_c = UPDATE_INTERVAL
    time_c = track["latLonTime"][index]
    while index < track.shape[0] - 1:
        time.sleep(UPDATE_INTERVAL)
        time_c += dt_c

        # Process each flight message when received
        if time_c >= track["latLonTime"][index + 1]:
            index = track["latLonTime"][time_c >= track["latLonTime"]].index[-1]
            flight_msg = make_flight_msg(track, index)
            logger.info(f"Publishing flight msg: {flight_msg}")
            handler.publish_to_topic(handler.flight_topic, json.dumps(flight_msg))

    # Read camera pointing file as a dataframe, and plot
    handler.camera_pointing_file.close()
    ts = pd.read_csv(handler.camera_pointing_filename)
    plot_time_series(ts)


if __name__ == "__main__":
    main()
