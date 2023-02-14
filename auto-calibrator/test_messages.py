import json
import logging
import math
from pathlib import Path
import time
import sys

sys.path.append(str(Path("../ptz-controller").expanduser()))
import MessageHandler
import make_handler

ALPHA_EXPECTED = 96.22945929035237
BETA_EXPECTED = 31.55893394983606
GAMMA_EXPECTED = 1.5230141040882903
HEARTBEAT_INTERVAL = 10.0
JPEG_RESOLUTION = "1920x1080"
UPDATE_INTERVAL = 0.01
MIN_ZOOM_EXPECTED = 0
MAX_ZOOM_EXPECTED = 9999

# Set precision of angle [deg] differences
PRECISION = 1.0e-5

logger = logging.getLogger("calibrator-integration")
logger.setLevel(logging.INFO)

def get_config_msg():
    """Populate a config message, by max and min horizontal
     and vertical FoV of camera.

    Parameters
    ----------
    None

    Returns
    -------
    msg : dict
        The configuration message
    """
    with open("data/config_msg_integration.json", "r") as f:
        msg = json.load(f)
    return msg


def get_pointing_error_msg():
    """Populate a pointing error message with test parameters.

    Parameters
    ----------
    None

    Returns
    -------
    msg : dict
        The pointing error message
    """
    with open("data/additional_info_msg.json", "r") as f:
        msg = json.load(f)
    return msg


def main():
    """Processes the config and pointing error messages using
    MQTT.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Make the calibrator, subscribe to all topics, and publish, or
    # process, one message to each topic
    logger.info("Making the calibrator, and subscribing to topics")
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

    # Publish messages
    config_msg = get_config_msg()
    pointing_error_msg = get_pointing_error_msg()
    logger.info(f"Publishing config msg: {config_msg}")
    handler.publish_to_topic(handler.config_topic, json.dumps(config_msg))
    time.sleep(UPDATE_INTERVAL)
    logger.info(f"Publishing pointing error msg: {pointing_error_msg}")
    calibrator.publish_to_topic(
        calibrator.pointing_error_topic, json.dumps(pointing_error_msg)
    )
    time.sleep(UPDATE_INTERVAL)

    assert math.fabs(calibrator.alpha - ALPHA_EXPECTED) < PRECISION
    assert math.fabs(calibrator.beta - BETA_EXPECTED) < PRECISION
    assert math.fabs(calibrator.gamma - GAMMA_EXPECTED) < PRECISION


if __name__ == "__main__":
    main()
