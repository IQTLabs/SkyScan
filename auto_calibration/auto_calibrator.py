import datetime
import json
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
sys.path.append(str(Path(os.getenv("EDGETECH_CORE_HOME")).expanduser()))
from base_mqtt_pub_sub import BaseMQTTPubSub

import utils_auto_calibrator


class AutoCalibrator(BaseMQTTPubSub):
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
        debug: bool = False,
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
        debug: bool
            Flag to debug the auto calibrator, or not


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
        self.debug = debug

        # Connect client on construction
        if not self.debug:
            self.connect_client()
            sleep(1)
            self.publish_registration("Auto Calibration Registration")

    def _calibration_callback(
        self: Any, _client: mqtt.Client, _userdata: Dict[Any, Any], payload: Any
    ) -> None:
        """
        Process calibration message, find tripod yaw, pitch, and roll that minimizes pointing error.
        Publish results for use by PTZ controller.

        Parameters
        ----------
        _client: mqtt.Client
            MQTT client
        _userdata: dict
            Any required user data
        payload: Dict
            A Dict with calibration information and a timestamp
        Returns
        -------
        None
        """

        # Decode calibration message
        # TODO: get correct msg format
        if self.debug:
            data = payload["data"]
        else:
            data = str(payload.payload.decode("utf-8"))["data"]

            # TODO: switch to logging
            print(
                "Received '{payload}' from `{topic}` topic".format(
                    payload=payload.payload.decode(), topic=payload.topic
                )
            )

        # Find tripod yaw, pitch, and roll that minimize pointing error
        rho_epsilon, tau_epsilon = self._calculate_calibration_error(data)
        self.alpha, self.beta, self.gamma = self._minimize_pointing_error(
            data, rho_epsilon, tau_epsilon
        )

        # Publish results to topic to be read by PTZ controller
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
        if not self.debug:
            self.publish_to_topic(self.publish_topic, json.dumps(publish_data))

        pass

    def _config_callback(
        self: Any, _client: mqtt.Client, _userdata: Dict[Any, Any], payload: Any
    ) -> None:
        """
        Process config message, update camera configuration.

        Parameters
        ----------
        _client: mqtt.Client
            MQTT client
        _userdata: dict
            Any required user data
        payload: Dict
            A Dict with config information and a timestamp
        Returns
        -------
        None
        """

        # Decode config message
        # TODO: get correct msg format
        if self.debug:
            data = payload["data"]
        else:
            data = str(payload.payload.decode("utf-8"))["data"]

            # TODO: switch to logging
            print(
                "Received '{payload}' from `{topic}` topic".format(
                    payload=payload.payload.decode(), topic=payload.topic
                )
            )

        # Set camera config values
        # Config message can include any or all values
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

        pass

    def _calculate_calibration_error(self, msg):

        """
        Calculate calibration error of camera using information from YOLO or equivalent bounding box.

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

        # FoV is calculated with 1 - zoom_percentage because max zoom indicates minimum FoV
        horizontal_fov = (
            (self.max_horizontal_fov - self.min_horizontal_fov) * (1 - zoom_percentage)
        ) + self.min_horizontal_fov
        vertical_fov = (
            (self.max_vertical_fov - self.min_vertical_fov) * (1 - zoom_percentage)
        ) + self.min_vertical_fov

        horizontal_degrees_per_pixel = horizontal_fov / self.horizontal_pixels
        vertical_degrees_per_pixel = vertical_fov / self.vertical_pixels

        # Get aircraft bounding box. Top left and bottom right points
        # Position is in pixels from the upper left corner of image, down and right
        bbox = msg["aircraft"]["bbox"]

        # Use bounding box information to calculate pan and tilt error in degrees

        center_x = (bbox[1] + bbox[3]) / 2
        center_y = (bbox[0] + bbox[2]) / 2

        # Positive values represents top and right, respectively
        horizontal_pixel_difference = center_x - (self.horizontal_pixels / 2)
        vertical_pixel_difference = (self.vertical_pixels / 2) - center_y

        rho_epsilon = horizontal_pixel_difference * horizontal_degrees_per_pixel
        tau_epsilon = vertical_pixel_difference * vertical_degrees_per_pixel

        return rho_epsilon, tau_epsilon

    @staticmethod
    def _calculate_pointing_error(
        alpha_beta_gamma,  # Independent vars
        msg,  # Parameters
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
            Represents first iteration of yaw pitch and roll [alpha, beta, gamma]
        msg: Dict
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

        # Compute position of the aircraft
        a_varphi = msg["aircraft"]["lat"]  # [deg]
        a_lambda = msg["aircraft"]["long"]  # [deg]
        a_h = msg["aircraft"]["altitude"]  # [m]
        r_XYZ_a = utils_auto_calibrator.compute_r_XYZ(a_lambda, a_varphi, a_h)

        # Compute position of the tripod
        t_varphi = msg["camera"]["lat"]  # [deg]
        t_lambda = msg["camera"]["long"]  # [deg]
        t_h = msg["camera"]["altitude"]  # [m]
        r_XYZ_t = utils_auto_calibrator.compute_r_XYZ(t_lambda, t_varphi, t_h)

        # Compute orthogonal transformation matrix from geocentric to
        # topocentric coordinates, and corresponding unit vectors
        # system of the tripod
        E_XYZ_to_ENz, e_E_XYZ, e_N_XYZ, e_z_XYZ = utils_auto_calibrator.compute_E(
            t_lambda, t_varphi
        )

        # Compute the rotations from the XYZ coordinate system to the uvw
        # (camera housing fixed) coordinate system
        alpha = alpha_beta_gamma[0]  # [deg]
        beta = alpha_beta_gamma[1]  # [deg]
        gamma = alpha_beta_gamma[2]  # [deg]
        _, _, _, E_XYZ_to_uvw, _, _, _ = utils_auto_calibrator.compute_rotations(
            e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, 0.0, 0.0
        )

        # Compute position in the uvw coordinate system of the aircraft
        # relative to the tripod
        r_uvw_a_t = np.matmul(E_XYZ_to_uvw, r_XYZ_a - r_XYZ_t)

        # Compute pan and tilt to point the camera at the aircraft given
        # the updated values of alpha, beta, and gamma
        rho = math.degrees(math.atan2(r_uvw_a_t[0], r_uvw_a_t[1]))  # [deg]
        tau = math.degrees(
            math.atan2(r_uvw_a_t[2], utils_auto_calibrator.norm(r_uvw_a_t[0:2]))
        )  # [deg]

        # Return pointing error
        return math.sqrt(
            (rho_0 + rho_epsilon - rho) ** 2 + (tau_0 + tau_epsilon - tau) ** 2
        )

    def _minimize_pointing_error(self, msg, rho_epsilon, tau_epsilon):
        """
        Find tripod yaw, pitch, and roll that minimizes pointing error.

        Parameters
        ----------
        msg: Dict
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
        rho_0 = msg["camera"]["pan"]
        tau_0 = msg["camera"]["tilt"]

        # Calculate alpha, beta, gamma that minimizes pointing error
        alpha_1, beta_1, gamma_1 = fmin_bfgs(
            self._calculate_pointing_error,
            x0,
            args=[msg, rho_0, tau_0, rho_epsilon, tau_epsilon],
        )

        return alpha_1, beta_1, gamma_1

    def main(self: Any) -> None:
        """
        Schedule heartbeat and subscribes to calibration and config topics with callbacks.

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
                sleep(0.001)

            except Exception as e:
                if self.debug:
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
    # call the main function
    template.main()
