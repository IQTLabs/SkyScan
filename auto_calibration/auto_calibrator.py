import datetime
import os
from pathlib import Path
import json
import math
from time import sleep
from typing import Any, Dict
import sys

import numpy as np
import paho.mqtt.client as mqtt
import schedule
from scipy.optimize import fmin_bfgs

# TODO: standardize method of importing base class
sys.path.append(str(Path(os.getenv("EDGETECH_CORE_HOME")).expanduser()))
from base_mqtt_pub_sub import BaseMQTTPubSub
import calibration_utils

# TODO: add logging

class AutoCalibrator(BaseMQTTPubSub):
    def __init__(
        self: Any,
        env_variable: Any,
        config_topic: str,
        calibration_topic: str,
        debug: bool = False,
        test: bool = False,
        **kwargs: Any,
    ):
        """Initialize the auto calibrator
                Parameters
                ----------
                config_topic: str
                    MQTT topic for subscribing to configuration messages
                calibration_topic: str
                    MQTT topic for subscribing to calibration messages
                debug: bool
                    Flag to debug the auto calibrator, or not
                test: bool
                    Flag to test the auto calibrator, or not
                Returns
                -------
                AutoCalibrator
                """
        # Parent class handles kwargs
        super().__init__(**kwargs)
        self.env_variable = env_variable
        self.calibration_topic = calibration_topic
        self.debug = debug
        self.test = test

        # Connect client on construction
        if not self.test:
            self.connect_client()
            sleep(1)
            self.publish_registration("Template Module Registration")

    def _calibration_callback(
        self: Any, _client: mqtt.Client, _userdata: Dict[Any, Any], payload: Any
    ) -> None:
        """
        Process calibration message.
        Parameters
        ----------
        _client: mqtt.Client
            MQTT client
        _userdata: dict
            Any required user data
        msg: Dict
            A Dict with calibration information
        Returns
        -------
        None
        """
        # Decode message:
        payload = json.loads(str(payload.payload.decode("utf-8")))

        print("Received '{payload}' from `{topic}` topic".format(
            payload=payload.payload.decode(), topic=payload.topic))

        # Get message from payload
        # TODO: get correct msg format
        msg = payload["data"]

        # calculate rho_0, tau_0, rho_epsilon and tau_epsilon
        rho_0, tau_0, rho_epsilon, tau_epsilon = self._calculate_calibration_error(payload)

        # Use bfgs minimize function to determine correct alpha beta and gamma
        alpha, beta, gamma = self._minimize(payload, rho_0, tau_0, rho_epsilon, tau_epsilon)

        # TODO: find correct format
        publish_data = {
            "timestamp": str(int(datetime.utcnow().timestamp())),
            "data": f"[{alpha}, {beta}, {gamma}]",
        }
        self.publish_to_topic(self.publish_topic, json.dumps(publish_data))

        pass

    @staticmethod
    def _calculate_calibration_error(msg):
        """
        Calculate calibration error of camera using information from YOLO or equivalent bounding box

        Parameters
        ----------
        msg: Dict
            A Dict with calibration information
        Returns
        -------
        rho_0 : int
            initial pan [degrees]
        tau_0 : int
            initial tilt [degrees]
        rho_epsilon : int
            pan error [degrees]
        tau_epsilon : int
            tilt error [degrees]
        """

        # Get values at time of message
        rho_0 = msg["pan"]
        tau_0 = msg["tilt"]

        # Throw error if camera not at expected zoom (9999)
        # TODO: Dynamic FoV for zoom
        zoom = msg["zoom"]
        if zoom != 9999:
            raise ValueError('Camera not at expected zoom. Auto-calibration failed')

        # Horizontal and vertical field of view for AXIS M5525–E PTZ Network Camera at max zoom (9999)
        horizontal_fov = 6.7  # in degrees
        vertical_fov = 3.8  # in degrees

        # Resolution of images is 1920x1080
        horizontal_pixels = 1920
        vertical_pixels = 1080

        horizontal_degrees_per_pixel = horizontal_fov / horizontal_pixels
        vertical_degrees_per_pixel = vertical_fov / vertical_pixels

        # Get aircraft bounding box. Position is in pixels from the upper left corner down and right
        bbox = msg["aircraft"]["bbox"]

        # Get horizontal and vertical centerpoints by averaging points of bbox
        horizontal_center = (bbox[1] + bbox[3]) / 2
        vertical_center = (bbox[0] + bbox[2]) / 2

        # Get pixel difference from center of bbox to center of image. Positive represents top and right, respectively
        horizontal_pixel_difference = horizontal_center - (horizontal_pixels / 2)
        vertical_pixel_difference = (vertical_pixels / 2) - vertical_center

        # Calculate difference in degrees vertical and horizontal
        rho_epsilon = horizontal_pixel_difference * horizontal_degrees_per_pixel
        tau_epsilon = vertical_pixel_difference * vertical_degrees_per_pixel

        return rho_0, tau_0, rho_epsilon, tau_epsilon

    @staticmethod
    def _calculate_pointing_error(alpha_beta_gamma,  # Independent vars
                                  msg, rho_0, tau_0, rho_epsilon, tau_epsilon):  # Parameters
        """
        Calculates pointing error with given parameters

        Parameters
        ----------
        alpha_beta_gamma: List
            A List with ints [alpha, beta, gamma]
        msg: Dict
            A Dict with calibration information
        rho_0 : int
            initial pan [degrees]
        tau_0 : int
            initial tilt [degrees]
        rho_epsilon : int
            pan error [degrees]
        tau_epsilon : int
            tilt error [degrees]
        Returns
        -------
        ___ : int
            pointing error
        """

        # Compute position of the aircraft
        a_varphi = msg["lat"]  # [deg]
        a_lambda = msg["lon"]  # [deg]
        a_h = msg["altitude"]  # [m]
        r_XYZ_a = calibration_utils.compute_r_XYZ(a_lambda, a_varphi, a_h)

        # Compute position of the tripod
        t_varphi = msg["camera"]["lat"]  # [deg]
        t_lambda = msg["camera"]["lon"]  # [deg]
        t_h = msg["camera"]["lon"]  # [m]
        r_XYZ_t = calibration_utils.compute_r_XYZ(t_lambda, t_varphi, t_h)

        # Compute orthogonal transformation matrix from geocentric to
        # topocentric coordinates, and corresponding unit vectors
        # system of the tripod
        E_XYZ_to_ENz, e_E_XYZ, e_N_XYZ, e_z_XYZ = calibration_utils.compute_E(t_lambda, t_varphi)

        # Compute the rotations from the XYZ coordinate system to the uvw
        # (camera housing fixed) coordinate system
        alpha = alpha_beta_gamma[0]  # [deg]
        beta = alpha_beta_gamma[1]  # [deg]
        gamma = alpha_beta_gamma[2]  # [deg]
        _, _, _, E_XYZ_to_uvw, _, _, _ = calibration_utils.compute_rotations(
            e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, 0.0, 0.0
        )

        # Compute position in the uvw coordinate system of the aircraft
        # relative to the tripod
        r_uvw_a_t = np.matmul(E_XYZ_to_uvw, r_XYZ_a - r_XYZ_t)

        # Compute pan and tilt to point the camera at the aircraft given
        # the updated values of alpha, beta, and gamma
        rho = math.degrees(math.atan2(r_uvw_a_t[0], r_uvw_a_t[1]))  # [deg]
        tau = math.degrees(
            math.atan2(r_uvw_a_t[2], calibration_utils.norm(r_uvw_a_t[0:2]))
        )  # [deg]

        # Return the pointing error to be minimized
        return math.sqrt((rho_0 + rho_epsilon - rho) ** 2 + (tau_0 + tau_epsilon - tau) ** 2)

    def _minimize(self, msg, rho_0, tau_0, rho_epsilon, tau_epsilon):
        """
        Runs the Broyden, Fletcher, Goldfarb, and Shanno minimization algorithm on pointing error method

        Parameters
        ----------
        msg: Dict
            A Dict with calibration information
        rho_0 : int
            initial pan [degrees]
        tau_0 : int
            initial tilt [degrees]
        rho_epsilon : int
            pan error [degrees]
        tau_epsilon : int
            tilt error [degrees]
        Returns
        -------
         alpha_1: int
            Yaw pointing error
         beta_1: int
            Pitch pointing error
         gamma_1: int
            Roll pointing error
        """

        # Initial guess
        alpha_0 = msg["tripod_yaw"]  # [deg]
        beta_0 = msg["tripod_pitch"]  # [deg]
        gamma_0 = msg["tripod_roll"]  # [deg]
        x0 = [alpha_0, beta_0, gamma_0]

        x1 = fmin_bfgs(self._calculate_pointing_error, x0, args=[msg, rho_0, tau_0, rho_epsilon, tau_epsilon])
        alpha_1 = x1[0]
        beta_1 = x1[1]
        gamma_1 = x1[2]

        return alpha_1, beta_1, gamma_1

    def main(self: Any) -> None:
        """
        Main function schedules heartbeat and subscribes to calibration topic with callback.

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
        env_variable=os.environ.get("ENV_VARIABLE"),
        calibration_topic=str(os.environ.get("CALIBRATION_TOPIC")),
        config_topic=os.environ.get("CONFIG_TOPIC"),
        mqtt_ip=os.environ.get("MQTT_IP"),
    )
    # call the main function
    template.main()