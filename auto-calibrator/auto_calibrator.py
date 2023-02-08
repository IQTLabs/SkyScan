import datetime
import json
import logging
import math
import os
import sys
from pathlib import Path
from time import sleep
from typing import Any, Dict

import numpy as np
import paho.mqtt.client as mqtt
import schedule
from scipy.optimize import fmin_bfgs

# TODO: standardize method of importing base class
sys.path.append(str(Path(os.getenv("CORE_PATH")).expanduser()))
from base_mqtt_pub_sub import BaseMQTTPubSub

# TODO: Huh?
sys.path.append(str(Path("../ptz-controller").expanduser()))
import ptz_utilities


root_logger = logging.getLogger()
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
root_logger.addHandler(ch)
root_logger.setLevel(logging.INFO)

logger = logging.getLogger("AutoCalibrator")
logger.setLevel(logging.INFO)


class AutoCalibrator(BaseMQTTPubSub):
    """TODO: Complete"""

    def __init__(
        self: Any,
        config_topic: str,
        calibration_topic: str,
        min_zoom,
        max_zoom,
        min_horizontal_fov,
        max_horizontal_fov,
        min_vertical_fov,
        max_vertical_fov,
        horizontal_pixels=1920,
        vertical_pixels=1080,
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        use_mqtt: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the auto calibrator.

        Parameters
        ----------
        config_topic: str
            MQTT topic for subscribing to configuration messages
        calibration_topic: str
            MQTT topic for subscribing to calibration messages
        min_zoom: int
            Minimum zoom setting on camera
        max_zoom: int
            Maximum zoom setting on camera
        min_horizontal_fov: float
            Camera horizontal FoV at maximum zoom
        max_horizontal_fov: float
            Camera horizontal FoV at minimum zoom
        min_vertical_fov: float
            Camera vertical FoV at maximum zoom
        horizontal_pixels: int
            Horizontal pixel count in image analyzed by bounding box
        vertical_pixels: int
            Vertical pixel count in image analyzed by bounding box
        alpha: float
            Tripod yaw
        beta: float
            Tripod pitch
        gamma: float
            Tripod roll
        use_mqtt: bool
            Flag to use MQTT, or not

        Returns
        -------
        AutoCalibrator
        """
        # Parent class handles kwargs
        super().__init__(**kwargs)
        self.calibration_topic = calibration_topic
        self.config_topic = config_topic
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.min_horizontal_fov = min_horizontal_fov
        self.max_horizontal_fov = max_horizontal_fov
        self.min_vertical_fov = min_vertical_fov
        self.max_vertical_fov = max_vertical_fov
        self.horizontal_pixels = horizontal_pixels
        self.vertical_pixels = vertical_pixels
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_mqtt = use_mqtt

        # Connect client on construction
        if self.use_mqtt:
            self.connect_client()
            sleep(1)
            self.publish_registration("Auto Calibration Registration")

    def _calibration_callback(
        self: Any, _client: mqtt.Client, _userdata: Dict[Any, Any], msg: Any
    ) -> None:
        """
        Process calibration message by finding tripod yaw, pitch, and
        roll that minimizes pointing error.  Publish results for use
        by PTZ controller.

        Parameters
        ----------
        _client: mqtt.Client
            MQTT client
        _userdata: dict
            Any required user data
        msg: Dict
            A Dict with calibration information and a timestamp

        Returns
        -------
        None
        """
        # Decode calibration message
        if self.use_mqtt:
            data = self.decode_payload(msg)

            # TODO: switch to logging
            print(
                "Received '{payload}' from `{topic}` topic".format(
                    payload=msg.payload.decode(), topic=msg.topic
                )
            )

        else:
            data = msg["data"]

        # Find tripod yaw, pitch, and roll that minimize pointing
        # error
        rho_epsilon, tau_epsilon = self._calculate_calibration_error(data)
        self.alpha, self.beta, self.gamma = self._minimize_pointing_error(
            data, rho_epsilon, tau_epsilon
        )

        # Publish results to calibration topic which is subscribed to
        # by PTZ controller
        publish_data = {
            "timestamp": str(int(datetime.datetime.utcnow().timestamp())),
            "data": {
                "camera": {
                    "tripod_yaw": self.alpha,
                    "tripod_pitch": self.beta,
                    "tripod_roll": self.gamma,
                }
            },
        }
        if self.use_mqtt:
            self.publish_to_topic(self.publish_topic, json.dumps(publish_data))

    def _config_callback(
        self: Any, _client: mqtt.Client, _userdata: Dict[Any, Any], msg: Any
    ) -> None:
        """Process config message, update camera configuration.

        Parameters
        ----------
        _client: mqtt.Client
            MQTT client
        _userdata: dict
            Any required user data
        msg: Dict
            A Dict with config information and a timestamp

        Returns
        -------
        None
        """
        # Decode config message
        if self.use_mqtt:
            data = self.decode_payload(msg)

            # TODO: switch to logging
            print(
                "Received '{payload}' from `{topic}` topic".format(
                    payload=msg.payload.decode(), topic=msg.topic
                )
            )

        else:
            data = msg["data"]

        # Set camera config values. Config message can include any or
        # all values.
        self.min_zoom = (
            data["camera"]["min_zoom"]
            if "min_zoom" in data["camera"]
            else self.min_zoom
        )
        self.max_zoom = (
            data["camera"]["max_zoom"]
            if "max_zoom" in data["camera"]
            else self.max_zoom
        )
        self.min_horizontal_fov = (
            data["camera"]["min_horizontal_fov"]
            if "min_horizontal_fov" in data["camera"]
            else self.min_horizontal_fov
        )
        self.max_horizontal_fov = (
            data["camera"]["max_horizontal_fov"]
            if "max_horizontal_fov" in data["camera"]
            else self.max_horizontal_fov
        )
        self.min_vertical_fov = (
            data["camera"]["min_vertical_fov"]
            if "min_vertical_fov" in data["camera"]
            else self.min_vertical_fov
        )
        self.max_vertical_fov = (
            data["camera"]["max_vertical_fov"]
            if "max_vertical_fov" in data["camera"]
            else self.max_vertical_fov
        )
        self.horizontal_pixels = (
            data["camera"]["horizontal_pixels"]
            if "horizontal_pixels" in data["camera"]
            else self.horizontal_pixels
        )
        self.vertical_pixels = (
            data["camera"]["vertical_pixels"]
            if "vertical_pixels" in data["camera"]
            else self.vertical_pixels
        )
        self.alpha = (
            data["camera"]["tripod_yaw"]
            if "tripod_yaw" in data["camera"]
            else self.alpha
        )
        self.beta = (
            data["camera"]["tripod_pitch"]
            if "tripod_pitch" in data["camera"]
            else self.beta
        )
        self.gamma = (
            data["camera"]["tripod_roll"]
            if "tripod_roll" in data["camera"]
            else self.gamma
        )

    def _calculate_calibration_error(self, msg):

        """
        Calculate calibration error of camera using information from
        YOLO or equivalent bounding box.

        Parameters
        ----------
        msg: Dict
            A Dict with calibration information

        Returns
        -------
        rho_epsilon : float
            Pan error [degrees]
        tau_epsilon : float
            Tilt error [degrees]
        """
        # Calculate FoV based on current zoom
        zoom = msg["camera"]["zoom"]
        zoom_percentage = (zoom - self.min_zoom) / (self.min_zoom + self.max_zoom)

        # FoV is calculated with 1 - zoom_percentage because max zoom
        # indicates minimum FoV
        horizontal_fov = (
            (self.max_horizontal_fov - self.min_horizontal_fov) * (1 - zoom_percentage)
        ) + self.min_horizontal_fov
        vertical_fov = (
            (self.max_vertical_fov - self.min_vertical_fov) * (1 - zoom_percentage)
        ) + self.min_vertical_fov
        horizontal_degrees_per_pixel = horizontal_fov / self.horizontal_pixels
        vertical_degrees_per_pixel = vertical_fov / self.vertical_pixels

        # Get aircraft bounding box: top left and bottom
        # right. Position is in pixels from the upper left corner of
        # image, down and right.
        bbox = msg["aircraft"]["bbox"]

        # Calculate pan and tilt error in degrees
        center_x = (bbox[1] + bbox[3]) / 2
        center_y = (bbox[0] + bbox[2]) / 2
        horizontal_pixel_difference = center_x - (
            self.horizontal_pixels / 2
        )  # Positive values represents top and right, respectively
        vertical_pixel_difference = (self.vertical_pixels / 2) - center_y
        rho_epsilon = horizontal_pixel_difference * horizontal_degrees_per_pixel
        tau_epsilon = vertical_pixel_difference * vertical_degrees_per_pixel

        return rho_epsilon, tau_epsilon

    @staticmethod
    def _calculate_pointing_error(
        alpha_beta_gamma,  # Independent vars
        data,  # Parameters
        rho_0,
        tau_0,
        rho_epsilon,
        tau_epsilon,
    ):
        """
        Calculates the pointing error with given yaw, pitch, and roll.

        Parameters
        ----------
        alpha_beta_gamma: List [int]
            Represents first iteration of yaw pitch and roll [alpha,
            beta, gamma]
        data: Dict
            A Dict with calibration information
        rho_0 : float
            Current pan [degrees]
        tau_0 : float
            Current tilt [degrees]
        rho_epsilon : float
            Pan error [degrees]
        tau_epsilon : float
            Tilt error [degrees]

        Returns
        -------
        ___ : float
            Pointing error with given yaw, pitch and roll
        """
        # Compute position of the aircraft in geocentric (XYZ)
        # coordinates
        a_varphi = data["aircraft"]["lat"]  # [deg]
        a_lambda = data["aircraft"]["long"]  # [deg]
        a_h = data["aircraft"]["altitude"]  # [m]
        r_XYZ_a = ptz_utilities.compute_r_XYZ(a_lambda, a_varphi, a_h)

        # Compute position of the tripod in geocentric (XYZ)
        # coordinates
        t_varphi = data["camera"]["lat"]  # [deg]
        t_lambda = data["camera"]["long"]  # [deg]
        t_h = data["camera"]["altitude"]  # [m]
        r_XYZ_t = ptz_utilities.compute_r_XYZ(t_lambda, t_varphi, t_h)

        # Compute orthogonal transformation matrix from geocentric
        # (XYZ) to topocentric (ENz) coordinates, and corresponding
        # topocentric unit vectors
        E_XYZ_to_ENz, e_E_XYZ, e_N_XYZ, e_z_XYZ = ptz_utilities.compute_E_XYZ_to_ENz(
            t_lambda, t_varphi
        )

        # Compute the rotations from the geocentric (XYZ) coordinate
        # system to the camera housing fixed (uvw) coordinate system
        alpha = alpha_beta_gamma[0]  # [deg]
        beta = alpha_beta_gamma[1]  # [deg]
        gamma = alpha_beta_gamma[2]  # [deg]
        _, _, _, E_XYZ_to_uvw, _, _, _ = ptz_utilities.compute_camera_rotations(
            e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, 0.0, 0.0
        )

        # Compute position in the camera housing fixed (uvw)
        # coordinate system of the aircraft relative to the tripod
        r_uvw_a_t = np.matmul(E_XYZ_to_uvw, r_XYZ_a - r_XYZ_t)

        # Compute pan and tilt to point the camera at the aircraft
        # given the updated values of alpha, beta, and gamma
        rho = math.degrees(math.atan2(r_uvw_a_t[0], r_uvw_a_t[1]))  # [deg]
        tau = math.degrees(
            math.atan2(r_uvw_a_t[2], ptz_utilities.norm(r_uvw_a_t[0:2]))
        )  # [deg]

        # Return pointing error
        return math.sqrt(
            (rho_0 + rho_epsilon - rho) ** 2 + (tau_0 + tau_epsilon - tau) ** 2
        )

    def _minimize_pointing_error(self, data, rho_epsilon, tau_epsilon):
        """
        Find tripod yaw, pitch, and roll that minimizes pointing error.

        Parameters
        ----------
        data: Dict
            A Dict with calibration information
        rho_epsilon : float
            Pan pointing error [degrees]
        tau_epsilon : float
            Tilt pointing error [degrees]

        Returns
        -------
        alpha_1: float
            Yaw that minimizes pointing error
        beta_1: float
            Pitch that minimizes pointing error
        gamma_1: float
            Roll that minimizes pointing error
        """
        # Get current yaw, pitch, roll for initial minimization guess
        alpha_0 = self.alpha  # [deg]
        beta_0 = self.beta  # [deg]
        gamma_0 = self.gamma  # [deg]
        x0 = [alpha_0, beta_0, gamma_0]

        # Get current pointing
        rho_0 = data["camera"]["pan"]
        tau_0 = data["camera"]["tilt"]

        # Calculate alpha, beta, gamma that minimizes pointing error
        alpha_1, beta_1, gamma_1 = fmin_bfgs(
            self._calculate_pointing_error,
            x0,
            args=[data, rho_0, tau_0, rho_epsilon, tau_epsilon],
        )

        return alpha_1, beta_1, gamma_1

    def main(self: Any) -> None:
        """
        Schedule heartbeat and subscribes to calibration and config
        topics with callbacks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Schedule heartbeat
        schedule.every(10).seconds.do(
            self.publish_heartbeat, payload="Template Module Heartbeat"
        )

        # Subscribe to calibration topic with callback
        self.add_subscribe_topic(self.calibration_topic, self._calibration_callback)

        # Subscribe to config topic with callback
        self.add_subscribe_topic(self.config_topic, self._config_callback)

        # Run
        while True:
            try:
                schedule.run_pending()
                # TODO: Make a parameter
                sleep(0.001)

            except Exception as e:
                # TODO: Use logging
                if self.use_mqtt:
                    print(e)


if __name__ == "__main__":
    # Instantiate auto calibrator and execute
    template = AutoCalibrator(
        calibration_topic=os.environ.get("CALIBRATION_TOPIC"),
        config_topic=os.environ.get("CONFIG_TOPIC"),
        min_zoom=int(os.environ.get("MIN_ZOOM")),
        max_zoom=int(os.environ.get("MAX_ZOOM")),
        min_horizontal_fov=float(os.environ.get("MIN_HORIZONTAL_FOV")),
        max_horizontal_fov=float(os.environ.get("MAX_HORIZONTAL_FOV")),
        min_vertical_fov=float(os.environ.get("MIN_VERTICAL_FOV")),
        max_vertical_fov=float(os.environ.get("MIN_HORIZONTAL_FOV")),
        mqtt_ip=os.environ.get("MQTT_IP"),
    )
    template.main()
