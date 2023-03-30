import datetime
from distutils.util import strtobool
import json
import logging
import math
import os
from pathlib import Path
from time import sleep
from typing import Any, Dict

import numpy as np
import paho.mqtt.client as mqtt
import schedule
from scipy.optimize import minimize, Bounds

from base_mqtt_pub_sub import BaseMQTTPubSub
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
    """Calibrate offset of tilt, pan, and zoom of the camera tripod
    and publish results.
    """

    def __init__(
        self: Any,
        config_topic: str,
        pointing_error_topic: str,
        calibration_topic: str,
        heartbeat_interval: float,
        min_zoom: int = 0,
        max_zoom: int = 9999,
        min_horizontal_fov_fit: float = 3.803123,
        max_horizontal_fov_fit: float = 48.31584,
        scale_horizontal_fov_fit: float = 0.001627997,
        horizontal_pixels: int = 1920,
        vertical_pixels: int = 1080,
        min_image_score: float = 0.925,
        max_bbox_area: float = 0.100,
        icao24: str = "NA",
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        use_mqtt: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize the auto calibrator.

        Parameters
        ----------
        config_topic: str
            MQTT topic for subscribing to configuration messages
        pointing_error_topic: str
            MQTT topic for subscribing to pointing error messages
        calibration_topic: str
            MQTT topic for subscribing to calibration messages
        heartbeat_interval: float
            Interval at which heartbeat message is to be published [s]
        min_zoom: int
            Minimum zoom setting on camera
        max_zoom: int
            Maximum zoom setting on camera
        max_horizontal_fov_fit: float
            Camera horizontal FoV at maximum zoom
        scale_horizontal_fov_fit: float
            Camera horizontal FoV scale
        min_horizontal_fov_fit: float
            Camera horizontal FoV at minimum zoom
        horizontal_pixels: int
            Horizontal pixel count in image analyzed by bounding box
        vertical_pixels: int
            Vertical pixel count in image analyzed by bounding box
        min_image_score: float
            Minimum accepted bbox score from image recognition to process aircraft
        max_bbox_area: float
            Maximum accepted area of bbox to process aircraft
        icao24: str
            icao24 code of most recently processed aircraft
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
        self.config_topic = config_topic
        self.pointing_error_topic = pointing_error_topic
        self.calibration_topic = calibration_topic
        self.heartbeat_interval = heartbeat_interval
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.min_horizontal_fov_fit = min_horizontal_fov_fit
        self.max_horizontal_fov_fit = max_horizontal_fov_fit
        self.scale_horizontal_fov_fit = scale_horizontal_fov_fit
        self.horizontal_pixels = horizontal_pixels
        self.vertical_pixels = vertical_pixels
        self.min_image_score = min_image_score
        self.max_bbox_area = max_bbox_area
        self.icao24 = icao24
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_mqtt = use_mqtt

        # Age of flight message
        self.flight_msg_age = 0.0  # [s]

        # List of pointing error data dictionaries and corresponding
        # pan and tilt errors
        self.data_list = []
        self.rho_epsilon_list = []
        self.tau_epsilon_list = []

        # Connect client on construction
        if self.use_mqtt:
            self.connect_client()
            sleep(1)
            self.publish_registration("Auto Calibration Registration")

        logger.info(
            f"""AutoCalibrator initialized with parameters:

    config_topic = {config_topic}
    pointing_error_topic = {pointing_error_topic}
    calibration_topic = {calibration_topic}
    heartbeat_interval = {heartbeat_interval}
    min_zoom = {min_zoom}
    max_zoom = {max_zoom}
    min_horizontal_fov_fit = {min_horizontal_fov_fit}
    max_horizontal_fov_fit = {max_horizontal_fov_fit}
    scale_horizontal_fov_fit = {scale_horizontal_fov_fit}
    horizontal_pixels = {horizontal_pixels}
    vertical_pixels = {vertical_pixels}
    min_image_score = {min_image_score}
    max_bbox_area = {max_bbox_area}
    icao24 = {icao24}
    alpha = {alpha}
    beta = {beta}
    gamma = {gamma}
    use_mqtt = {use_mqtt}
            """
        )

    def decode_payload(self, payload):
        """
        Decode the payload carried by a message.

        Parameters
        ----------
        payload: Any
            A JSON string with {timestamp: ____, data: ____,}

        Returns
        -------
        data : dict
            The data component of the payload
        """
        # TODO: Establish and use message format convention
        content = json.loads(str(payload.decode("utf-8")))
        if "data" in content:
            data = content["data"]
        else:
            data = content
        return data

    def _pointing_error_callback(
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
        # Decode pointing error message
        if self.use_mqtt:
            data = self.decode_payload(msg.payload)
        else:
            data = msg
        logger.info(f"Received: {data}, from topic: {self.pointing_error_topic}")

        # Process messages describing images with sufficiently large
        # score and small size. Note that the bbox provides the screen
        # coordinates of the upper left and lower right corners.
        icao24 = Path(data["imagefile"]).stem.split("_")[0]
        bbox_data = data["aircraft"]["bbox"][0]
        bbox = bbox_data["bbox"]
        image_area = self.horizontal_pixels * self.vertical_pixels
        bbox_area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / image_area
        score = bbox_data["score"]
        if (
            score < self.min_image_score
            or bbox_area > self.max_bbox_area
            # or icao24 == self.icao24  # Include this test to calibrate once per aircraft
        ):
            logger.info(
                f"Skipping aircraft: {icao24}, with bbox area: {bbox_area}, and score: {score}"
            )
            return
        else:
            logger.info(
                f"Processing aircraft: {icao24}, with bbox area: {bbox_area}, and score: {score}"
            )
            self.icao24 = icao24

        # Accumulate pointing error data dictionaries and
        # corresponding pan and tilt errors
        self.data_list.append(data)
        rho_epsilon, tau_epsilon = self._calculate_calibration_error(data)
        self.rho_epsilon_list.append(rho_epsilon)
        self.tau_epsilon_list.append(tau_epsilon)
        if len(self.data_list) < 8:
            return

        # Find tripod yaw, pitch, roll and lead time that minimize
        # pointing error.
        alpha, beta, gamma = self._minimize_pointing_error()

        # Clear pointing error data dictionaries and corresponding pan
        # and tilt errors
        self.data_list = []
        self.rho_epsilon_list = []
        self.tau_epsilon_list = []

        # Assign an exponentially weighted average for the current
        # yaw, pitch, roll, and lead time
        # TODO: Review weighting scheme
        self.alpha = (self.alpha + alpha) / 2.0
        self.beta = (self.beta + beta) / 2.0
        self.gamma = (self.gamma + gamma) / 2.0

        # Publish results to calibration topic which is subscribed to
        # by PTZ controller
        calibration = {
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
            self.publish_to_topic(self.calibration_topic, json.dumps(calibration))
            logger.info(f"Published: {calibration}, to topic: {self.calibration_topic}")

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
            data = self.decode_payload(msg.payload)
        else:
            data = msg["data"]
        logger.info(f"Received: {data}, from topic: {self.config_topic}")

        # Set camera config values. Config message can include any or
        # all values
        # TODO: Update with current values
        if "min_zoom" in data["camera"]:
            old_min_zoom = self.min_zoom
            self.min_zoom = data["camera"]["min_zoom"]
            logger.info(
                f"Configuration min_zoom updated from {old_min_zoom} to {self.min_zoom}"
            )
        if "max_zoom" in data["camera"]:
            old_max_zoom = self.max_zoom
            self.max_zoom = data["camera"]["max_zoom"]
            logger.info(
                f"Configuration max_zoom updated from {old_max_zoom} to {self.max_zoom}"
            )
        if "min_horizontal_fov_fit" in data["camera"]:
            old_min_horizontal_fov_fit = self.min_horizontal_fov_fit
            self.min_horizontal_fov_fit = data["camera"]["min_horizontal_fov_fit"]
            logger.info(
                f"Configuration min_horizontal_fov_fit updated from {old_min_horizontal_fov_fit} to {self.min_horizontal_fov_fit} "
            )
        if "max_horizontal_fov_fit" in data["camera"]:
            old_max_horizontal_fov_fit = self.max_horizontal_fov_fit
            self.max_horizontal_fov_fit = data["camera"]["max_horizontal_fov_fit"]
            logger.info(
                f"Configuration max_horizontal_fov_fit updated from {old_max_horizontal_fov_fit} to {self.max_horizontal_fov_fit} "
            )
        if "scale_horizontal_fov_fit" in data["camera"]:
            old_scale_horizontal_fov_fit = self.scale_horizontal_fov_fit
            self.scale_horizontal_fov_fit = data["camera"]["scale_horizontal_fov_fit"]
            logger.info(
                f"Configuration scale_horizontal_fov_fit updated from {old_scale_horizontal_fov_fit} to {self.scale_horizontal_fov_fit} "
            )
        if "horizontal_pixels" in data["camera"]:
            old_horizontal_pixels = self.horizontal_pixels
            self.horizontal_pixels = data["camera"]["horizontal_pixels"]
            logger.info(
                f"Configuration horizontal_pixels updated from {old_horizontal_pixels} to {self.horizontal_pixels}"
            )
        if "vertical_pixels" in data["camera"]:
            old_vertical_pixels = self.vertical_pixels
            self.vertical_pixels = data["camera"]["vertical_pixels"]
            logger.info(
                f"Configuration vertical_pixels updated from {old_vertical_pixels} to {self.vertical_pixels}"
            )
        if "min_image_score" in data["camera"]:
            old_min_image_score = self.min_image_score
            self.min_image_score = data["camera"]["min_image_score"]
            logger.info(
                f"Configuration min_image_score updated from {old_min_image_score} to {self.min_image_score}"
            )
        if "max_bbox_area" in data["camera"]:
            old_max_bbox_area = self.max_bbox_area
            self.max_bbox_area = data["camera"]["max_bbox_area"]
            logger.info(
                f"Configuration max_bbox_area updated from {old_max_bbox_area} to {self.max_bbox_area}"
            )
        if "tripod_yaw" in data["camera"]:
            old_alpha = self.alpha
            self.alpha = data["camera"]["tripod_yaw"]
            logger.info(
                f"Configuration tripod_yaw updated from {old_alpha} to {self.alpha}"
            )
        if "tripod_pitch" in data["camera"]:
            old_beta = self.beta
            self.beta = data["camera"]["tripod_pitch"]
            logger.info(
                f"Configuration tripod_pitch updated from {old_beta} to {self.beta}"
            )
        if "tripod_roll" in data["camera"]:
            old_gamma = self.gamma
            self.gamma = data["camera"]["tripod_roll"]
            logger.info(
                f"Configuration tripod_roll updated from {old_gamma} to {self.gamma}"
            )

    def _calculate_calibration_error(self, data):
        """Calculate calibration error of camera using information
        from YOLO or equivalent bounding box.

        Parameters
        ----------
        data: Dict
            A Dict with calibration information

        Returns
        -------
        rho_epsilon : float
            Pan error [deg]
        tau_epsilon : float
            Tilt error [deg]
        """
        # Calculate FoV based on current zoom
        zoom = data["camera"]["zoom"]

        # Calculate horizontal and vertical fov based on exponential
        # FoV fit and aspect ratio
        fit_horizontal_fov = (
            self.max_horizontal_fov_fit * math.exp(-self.scale_horizontal_fov_fit * zoom)
            + self.min_horizontal_fov_fit
        )
        aspect_ratio = self.vertical_pixels / self.horizontal_pixels
        fit_vertical_fov = math.degrees(
            math.atan(math.tan(math.radians(fit_horizontal_fov)) * aspect_ratio)
        )
        horizontal_degrees_per_pixel = fit_horizontal_fov / self.horizontal_pixels
        vertical_degrees_per_pixel = fit_vertical_fov / self.vertical_pixels

        # Get aircraft bounding box. Note that the bbox provides the
        # screen coordinates of the upper left and lower right
        # corners, with position in pixels from the upper left corner
        # of image, down and right.
        # TODO: Establish and use message format convention
        bbox = data["aircraft"]["bbox"][0]["bbox"]
        logger.info(f"Got bbox: {bbox}")

        # Calculate pan and tilt error in degrees. Note that positive
        # values are toward right and top, respectively
        horizontal_box_center = (bbox[0] + bbox[2]) / 2
        vertical_box_center = (bbox[1] + bbox[3]) / 2
        horizontal_pixel_difference = horizontal_box_center - self.horizontal_pixels / 2
        vertical_pixel_difference = self.vertical_pixels / 2 - vertical_box_center
        rho_epsilon = horizontal_pixel_difference * horizontal_degrees_per_pixel
        tau_epsilon = vertical_pixel_difference * vertical_degrees_per_pixel
        logger.info(f"Found rho_epsilon: {rho_epsilon}, tau_epsilon: {tau_epsilon} [deg]")

        return rho_epsilon, tau_epsilon

    @staticmethod
    def _calculate_pointing_error(
        parameters,
        data_list,
        rho_epsilon_list,
        tau_epsilon_list,
    ):
        """Calculates the pointing error with given yaw, pitch, roll,
        and lead time.

        Parameters
        ----------
        parameters: List [int]
            Minimization parameters: yaw, pitch, roll, [deg], and lead
            time [s]
        data_list: List [Dict]
            A list of dicts with calibration information
        rho_epsilon_list : List [float]
            A list of pan errors [degrees]
        tau_epsilon : List [float]
            A list of tilt errors [degrees]

        Returns
        -------
        ___ : float
            Pointing error with given yaw, pitch, roll, and lead time
        """
        # Consider each element in the data, and pointing error lists
        pointing_error = 0
        for idx in range(len(data_list)):
            data = data_list[idx]
            rho_epsilon = rho_epsilon_list[idx]
            tau_epsilon = tau_epsilon_list[idx]

            # Assign camera pan and tilt
            rho_c = data["camera"]["rho_c"]
            tau_c = data["camera"]["tau_c"]

            # Compute orthogonal transformation matrix from geocentric
            # (XYZ) to topocentric (ENz) coordinates, and corresponding
            # topocentric unit vectors
            lambda_t = data["camera"]["lambda_t"]  # [deg]
            varphi_t = data["camera"]["varphi_t"]  # [deg]
            E_XYZ_to_ENz, e_E_XYZ, e_N_XYZ, e_z_XYZ = ptz_utilities.compute_E_XYZ_to_ENz(
                lambda_t, varphi_t
            )

            # Compute the rotations from the geocentric (XYZ) coordinate
            # system to the camera housing fixed (uvw) coordinate system
            alpha = parameters[0]  # [deg]
            beta = parameters[1]  # [deg]
            gamma = parameters[2]  # [deg]
            _, _, _, E_XYZ_to_uvw, _, _, _ = ptz_utilities.compute_camera_rotations(
                e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, 0.0, 0.0
            )

            # TODO: Remove?
            # Compute position in the topocentric (ENz) coordinate system
            # of the aircraft relative to the tripod at time one
            r_ENz_a_1_t = (
                np.array(data["aircraft"]["r_ENz_a_0_t"])
                # + np.array(data["aircraft"]["v_ENz_a_0_t"]) * (
                #     data["aircraft"]["lead_time"] + data["aircraft"]["flight_msg_age"]
                # )
            )

            # Compute position, at time one, in the geocentric (XYZ)
            # coordinate system of the aircraft relative to the tripod
            r_XYZ_a_1_t = np.matmul(E_XYZ_to_ENz.transpose(), r_ENz_a_1_t)

            # Compute position in the camera housing fixed (uvw)
            # coordinate system of the aircraft relative to the tripod
            r_uvw_a_1_t = np.matmul(E_XYZ_to_uvw, r_XYZ_a_1_t)

            # Compute pan and tilt to point the camera at the aircraft
            # given the updated values of alpha, beta, and gamma
            rho_a = math.degrees(math.atan2(r_uvw_a_1_t[0], r_uvw_a_1_t[1]))  # [deg]
            tau_a = math.degrees(
                math.atan2(r_uvw_a_1_t[2], ptz_utilities.norm(r_uvw_a_1_t[0:2]))
            )  # [deg]

            # Accumulate pointing error. Note that rho_c + rho_epsilon
            # gives the measured pan required to point at the
            # aircraft, while rho_a gives the pan required to point at
            # the aircraft with updated yaw, pitch, roll, and lead
            # time. As a result, the updated camera yaw, pitch, roll,
            # and lead time that minimizes the difference will allow
            # the pan required to point at the aircraft to be computed
            # with minimum error. Of course, the same comment applies
            # for tilt.
            pointing_error += math.sqrt(
                (rho_c + rho_epsilon - rho_a) ** 2 + (tau_c + tau_epsilon - tau_a) ** 2
            )

        return pointing_error

    def _minimize_pointing_error(self):
        """Find tripod yaw, pitch, roll, and lead time that minimizes
        pointing error.

        Parameters
        ----------
        None

        Returns
        -------
        alpha: float
            Yaw that minimizes pointing error
        beta: float
            Pitch that minimizes pointing error
        gamma: float
            Roll that minimizes pointing error
        """
        # Use current yaw, pitch, roll, and lead time for initial
        # minimization guess
        x0 = [self.alpha, self.beta, self.gamma]

        # Calculate alpha, beta, gamma, and lead time that minimizes
        # pointing error
        # TODO: Make bounds a parameter?
        res = minimize(
            self._calculate_pointing_error,
            x0,
            args=(self.data_list, self.rho_epsilon_list, self.tau_epsilon_list),
            bounds=Bounds(
                lb=[-0.5, -0.5, -0.5],
                ub=[0.5, 0.5, 0.5]
            )
        )
        if res.success:
            alpha = res.x[0]
            beta = res.x[1]
            gamma = res.x[2]
            logger.info(
                f"Minimization gives updated alpha: {alpha}, beta: {beta}, and gamma: {gamma}"
            )
        else:
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
            logger.info(
                f"Minimization failed, using original alpha: {alpha}, beta: {beta}, and gamma: {gamma}"
            )
        return alpha, beta, gamma

    def main(self: Any) -> None:
        """Schedule heartbeat and subscribes to calibration and config
        topics with callbacks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Schedule heartbeat
        schedule.every(self.heartbeat_interval).seconds.do(
            self.publish_heartbeat, payload="Auto-Calibrator Module Heartbeat"
        )

        # Subscribe to config topic with callback
        self.add_subscribe_topic(self.config_topic, self._config_callback)

        # Subscribe to pointing error topic with callback
        self.add_subscribe_topic(
            self.pointing_error_topic, self._pointing_error_callback
        )

        # Run
        while True:
            try:
                schedule.run_pending()
                # TODO: Make a parameter
                sleep(0.001)

            except Exception as e:
                if self.use_mqtt:
                    logger.error(e)


if __name__ == "__main__":
    # Instantiate auto calibrator and execute
    auto_calibrator = AutoCalibrator(
        mqtt_ip=os.getenv("MQTT_IP"),
        config_topic=os.getenv("CONFIG_TOPIC"),
        pointing_error_topic=os.getenv("POINTING_ERROR_TOPIC"),
        calibration_topic=os.getenv("CALIBRATION_TOPIC"),
        heartbeat_interval=float(os.getenv("HEARTBEAT_INTERVAL")),
        min_zoom=int(os.getenv("MIN_ZOOM")),
        max_zoom=int(os.getenv("MAX_ZOOM")),
        min_horizontal_fov_fit=float(os.getenv("MIN_HORIZONTAL_FOV_FIT")),
        max_horizontal_fov_fit=float(os.getenv("MAX_HORIZONTAL_FOV_FIT")),
        scale_horizontal_fov_fit=float(os.getenv("SCALE_HORIZONTAL_FOV_FIT")),
        horizontal_pixels=int(os.getenv("HORIZONTAL_PIXELS")),
        vertical_pixels=int(os.getenv("VERTICAL_PIXELS")),
        min_image_score=float(os.getenv("MIN_IMAGE_SCORE")),
        max_bbox_area=float(os.getenv("MAX_BBOX_AREA")),
        use_mqtt=strtobool(os.getenv("USE_MQTT")),
    )
    auto_calibrator.main()
