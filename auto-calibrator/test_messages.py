import json
import logging
import math
import time
from typing import Any, Dict

import paho.mqtt.client as mqtt

from base_mqtt_pub_sub import BaseMQTTPubSub

UPDATE_INTERVAL = 0.1

logger = logging.getLogger("calibrator-messages")
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


class MessageHandler(BaseMQTTPubSub):
    """Subscribe to all required topics."""

    def __init__(
        self: Any,
        config_topic: str,
        pointing_error_topic: str,
        calibration_topic: str,
        **kwargs: Any,
    ):
        """Initialize a MessageHandler by subscribing to all required
        topics, connecting to the MQTT broker.

        Parameters
        ----------
        config_topic: str
            MQTT topic for publishing or subscribing to configuration
            messages
        pointing_error_topic: str
            MQTT topic for publishing or subscribing to pointing error
            messages
        calibration_topic: str
            MQTT topic for publishing or subscribing to calibration
            messages

        Returns
        -------
        MessageHandler
        """
        # Parent class handles kwargs, including MQTT IP
        super().__init__(**kwargs)
        self.config_topic = config_topic
        self.pointing_error_topic = pointing_error_topic
        self.calibration_topic = calibration_topic

        # Connect MQTT client
        logger.info("Connecting MQTT client")
        self.connect_client()
        time.sleep(5)
        self.publish_registration("Message Handler Module Registration")

    def _calibration_callback(
            self: Any, _client: mqtt.Client, _userdata: Dict[Any, Any], msg: Any
    ) -> None:
        """Test content of calibration message.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        data = self.decode_payload(msg.payload)
        alpha_expected = 96.22945929035237
        beta_expected = 31.55893394983606
        gamma_expected = 1.5230141040882903
        precision = 1.0e-5

        assert math.fabs(data["camera"]["tripod_yaw"] - alpha_expected) < precision
        assert math.fabs(data["camera"]["tripod_pitch"] - beta_expected) < precision
        assert math.fabs(data["camera"]["tripod_roll"] - gamma_expected) < precision
        logger.info("Calibration successful")


def make_handler():
    """Construct a MessageHandler.

    Note that an MQTT broker must be started manually.

    Parameters
    ----------
    None

    Returns
    -------
    handler: MessageHandler
        The message handler
    """
    handler = MessageHandler(
        mqtt_ip="127.0.0.1",
        config_topic="skyscan/config/json",
        pointing_error_topic="skyscan/pointing_error/json",
        calibration_topic="skyscan/calibration/json",
    )
    return handler

def get_config_msg():
    """Load mock config message."""
    with open("data/config_msg_integration.json") as f:
        msg = json.load(f)
    return msg

def get_pointing_error_msg():
    """Load mock calibration message."""
    with open("data/additional_info_msg.json") as f:
        msg = json.load(f)
    return msg


def main():
    """Publish config and pointing error messages, and subscribe
    to calibration topic for testing.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Make the handler, and subscribe to the logger topic
    logger.info("Making the handler, and subscribing to topics")
    handler = make_handler()
    handler.add_subscribe_topic(handler.calibration_topic, handler._calibration_callback)

    # Publish the configuration and pointing error message
    config_msg = get_config_msg()
    pointing_error_msg = get_pointing_error_msg()
    logger.info(f"Publishing config msg: {config_msg}")
    handler.publish_to_topic(handler.config_topic, json.dumps(config_msg))
    time.sleep(UPDATE_INTERVAL)
    logger.info(f"Publishing pointing_error msg: {pointing_error_msg}")
    handler.publish_to_topic(handler.pointing_error_topic, json.dumps(pointing_error_msg))
    time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    main()
