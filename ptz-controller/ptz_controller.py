from datetime import datetime
from distutils.util import strtobool
import json
import logging
import math
import os
from pathlib import Path
from time import sleep, time
from typing import Any, Dict
import sys

import numpy as np
import quaternion
import paho.mqtt.client as mqtt
import schedule
from sensecam_control import vapix_config, vapix_control

# TODO: Agree on a method for importing the base class
sys.path.append(str(Path(os.getenv("CORE_PATH")).expanduser()))
from base_mqtt_pub_sub import BaseMQTTPubSub
import ptz_utilities

root_logger = logging.getLogger()
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
root_logger.addHandler(ch)

logger = logging.getLogger("ptz-controller")
logger.setLevel(logging.INFO)


class PtzController(BaseMQTTPubSub):
    """Point the camera at the aircraft using a proportional rate
    controller, and capture intervals while in track."""

    def __init__(
        self: Any,
        camera_ip: str,
        camera_user: str,
        camera_password: str,
        config_topic: str,
        calibration_topic: str,
        flight_topic: str,
        logger_topic: str,
        heartbeat_interval: float,
        update_interval: float,
        capture_interval: float,
        lead_time: float,
        pan_gain: float,
        pan_rate_min: float,
        pan_rate_max: float,
        tilt_gain: float,
        tilt_rate_min: float,
        tilt_rate_max: float,
        jpeg_resolution: str,
        jpeg_compression: int,
        use_mqtt: bool = True,
        use_camera: bool = True,
        log_to_mqtt: bool = False,
        **kwargs: Any,
    ):
        """Instanstiate the PTZ controller by connecting to the camera
        and message broker, and initializing data attributes.

        Parameters
        ----------
        camera_ip: str
            Camera IP address
        camera_user: str
            Camera user name
        camera_password: str
            Camera user password
        config_topic: str
            MQTT topic for subscribing to configuration messages
        calibration_topic: str
            MQTT topic for subscribing to calibration messages
        flight_topic: str
            MQTT topic for subscribing to flight messages
        logger_topic: str
            MQTT topic for publishing or subscribing to logger messages
        heartbeat_interval: float
            Interval at which heartbeat message is to be published [s]
        update_interval: float
            Interval at which pointing of the camera is computed [s]
        capture_interval: float
            Interval at which the camera image is captured [s]
        lead_time: float
            Lead time used when computing camera pointing to the
            aircraft [s]
        pan_gain: float
            Proportional control gain for pan error [1/s]
        pan_rate_min: float
            Camera pan rate minimum [deg/s]
        pan_rate_max: float
            Camera pan rate maximum [deg/s]
        tilt_gain: float
            Proportional control gain for tilt error [1/s]
        tilt_rate_min: float
            Camera tilt rate minimum [deg/s]
        tilt_rate_max: float
            Camera tilt rate maximum [deg/s]
        jpeg_resolution: str
            Image capture resolution, for example, "1920x1080"
        jpeg_compression: int
            Image compression: 0 to 100
        use_mqtt: bool
            Flag to use MQTT, or not
        use_camera: bool
            Flag to use camera configuration and control, or not
        log_to_mqtt: bool
            Flag to publish logger messages to MQTT, or not

        Returns
        -------
        PtzController
        """
        # Parent class handles kwargs, including MQTT IP
        super().__init__(**kwargs)
        self.camera_ip = camera_ip
        self.camera_user = camera_user
        self.camera_password = camera_password
        self.config_topic = config_topic
        self.calibration_topic = calibration_topic
        self.flight_topic = flight_topic
        self.logger_topic = logger_topic
        self.heartbeat_interval = heartbeat_interval
        self.update_interval = update_interval
        self.capture_interval = capture_interval
        self.lead_time = lead_time
        self.pan_gain = pan_gain
        self.pan_rate_min = pan_rate_min
        self.pan_rate_max = pan_rate_max
        self.tilt_gain = tilt_gain
        self.tilt_rate_min = tilt_rate_min
        self.tilt_rate_max = tilt_rate_max
        self.jpeg_resolution = jpeg_resolution
        self.jpeg_compression = jpeg_compression
        self.use_mqtt = use_mqtt
        self.use_camera = use_camera
        self.log_to_mqtt = log_to_mqtt

        # Construct camera configuration and control
        if self.use_camera:
            logger.info("Constructing camera configuration and control")
            self.camera_configuration = vapix_config.CameraConfiguration(
                self.camera_ip, self.camera_user, self.camera_password
            )
            self.camera_control = vapix_control.CameraControl(
                self.camera_ip, self.camera_user, self.camera_password
            )
        else:
            self.camera_configuration = None
            self.camera_control = None

        # Connect MQTT client
        if self.use_mqtt:
            logger.info("Connecting MQTT client")
            self.connect_client()
            sleep(1)
            self.publish_registration("PTZ Controller Module Registration")

        # Tripod longitude, latitude, and altitude
        self.lambda_t = 0.0  # [deg]
        self.varphi_t = 0.0  # [deg]
        self.h_t = 0.0  # [m]

        # East, North, and zenith unit vectors
        self.e_E_XYZ = None
        self.e_N_XYZ = None
        self.e_z_XYZ = None

        # Orthogonal transformation matrix from geocentric to
        # topocentric coordinates
        self.E_XYZ_to_ENz = None

        # Tripod position in the geocentric coordinate system
        self.r_XYZ_t = None

        # Aircraft identifier, time of flight message and
        # corresponding aircraft position and velocity
        self.icao24 = "NA"
        self.time_a = 0.0  # [s]
        self.r_rst_a_0_t = None  # [m/s]
        self.v_rst_a_0_t = None  # [m/s]

        # Tripod yaw, pitch, and roll angles
        self.alpha = 0.0  # [deg]
        self.beta = 0.0  # [deg]
        self.gamma = 0.0  # [deg]

        # Tripod yaw, pitch, and roll rotation quaternions
        self.q_alpha = None
        self.q_beta = None
        self.q_gamma = None

        # Orthogonal transformation matrix from geocentric (XYZ) to
        # camera housing fixed (uvw) coordinates
        self.E_XYZ_to_uvw = None

        # Aircraft pan and tilt angles
        self.rho_a = 0.0  # [deg]
        self.tau_a = 0.0  # [deg]

        # Pan, and tilt rotation quaternions
        self.q_row = None
        self.q_tau = None

        # Orthogonal transformation matrix from camera housing (uvw)
        # to camera fixed (rst) coordinates
        self.E_XYZ_to_rst = None

        # Aircraft pan and tilt rates
        self.rho_dot_a = 0.0  # [deg/s]
        self.tau_dot_a = 0.0  # [deg/s]

        # Time of pointing update, camera pan and tilt angles, and
        # zoom
        self.time_c = 0.0  # [s]
        self.rho_c = 0.0  # [deg]
        self.tau_c = 0.0  # [deg]
        self.zoom = 0  # -100 to 100 [-]

        # Camera pan and tilt rates
        self.rho_dot_c = 0.0  # [deg/s]
        self.tau_dot_c = 0.0  # [deg/s]

        # Camera pan and tilt rate differences
        self.delta_rho_dot_c = 0.0  # [deg/s]
        self.delta_tau_dot_c = 0.0  # [deg/s]

        # Capture boolean and last capture time
        self.do_capture = False
        self.capture_time = 0.0

    def _config_callback(
        self: Any, _client: mqtt.Client, _userdata: Dict[Any, Any], msg: Any
    ) -> None:
        """
        Process configuration message.

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
        # Assign position of the tripod
        if self.use_mqtt:
            data = self.decode_payload(msg)
        else:
            data = msg["data"]
        logger.info(f"Processing config msg data: {data}")
        self.lambda_t = data["tripod_longitude"]  # [deg]
        self.varphi_t = data["tripod_latitude"]  # [deg]
        self.h_t = data["tripod_altitude"]  # [m]
        # TODO: Set other values?

        # Compute tripod position in the geocentric (XYZ) coordinate
        # system
        self.r_XYZ_t = ptz_utilities.compute_r_XYZ(
            self.lambda_t, self.varphi_t, self.h_t
        )

        # Compute orthogonal transformation matrix from geocentric
        # (XYZ) to topocentric (ENz) coordinates
        (
            self.E_XYZ_to_ENz,
            self.e_E_XYZ,
            self.e_N_XYZ,
            self.e_z_XYZ,
        ) = ptz_utilities.compute_E_XYZ_to_ENz(self.lambda_t, self.varphi_t)

    def _calibration_callback(
        self: Any, _client: mqtt.Client, _userdata: Dict[Any, Any], msg: Any
    ) -> None:
        """
        Process calibration message.

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
        # Assign camera housing rotation angles
        if self.use_mqtt:
            data = self.decode_payload(msg)
        else:
            data = msg["data"]
        logger.info(f"Processing calibration msg data: {data}")
        self.alpha = data["camera"]["tripod_yaw"]  # [deg]
        self.beta = data["camera"]["tripod_pitch"]  # [deg]
        self.gamma = data["camera"]["tripod_roll"]  # [deg]

        # Compute the rotations from the geocentric (XYZ) coordinate
        # system to the camera housing fixed (uvw) coordinate system
        (
            self.q_alpha,
            self.q_beta,
            self.q_gamma,
            self.E_XYZ_to_uvw,
            _,
            _,
            _,
        ) = ptz_utilities.compute_camera_rotations(
            self.e_E_XYZ,
            self.e_N_XYZ,
            self.e_z_XYZ,
            self.alpha,
            self.beta,
            self.gamma,
            self.rho_c,
            self.tau_c,
        )

    def _flight_callback(
        self: Any, _client: mqtt.Client, _userdata: Dict[Any, Any], msg: Any
    ) -> None:
        """
        Process flight message.

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
        # Assign position and velocity of the aircraft
        if self.use_mqtt:
            data = self.decode_payload(msg)
        else:
            data = msg["data"]
        logger.info(f"Processing flight msg data: {data}")
        self.icao24 = data["icao24"]
        self.time_a = data["latLonTime"]  # [s]
        self.time_c = self.time_a
        lambda_a = data["lon"]  # [deg]
        varphi_a = data["lat"]  # [deg]
        h_a = data["altitude"]  # [m]
        track_a = data["track"]  # [deg]
        ground_speed_a = data["groundSpeed"]  # [m/s]
        vertical_rate_a = data["verticalRate"]  # [m/s]

        # Compute position in the geocentric (XYZ) coordinate system
        # of the aircraft relative to the tripod at time zero, the
        # observation time
        r_XYZ_a_0 = ptz_utilities.compute_r_XYZ(lambda_a, varphi_a, h_a)
        r_XYZ_a_0_t = r_XYZ_a_0 - self.r_XYZ_t

        # Compute position and velocity in the topocentric (ENz)
        # coordinate system of the aircraft relative to the tripod at
        # time zero, and position at slightly later time one
        r_ENz_a_0_t = np.matmul(self.E_XYZ_to_ENz, r_XYZ_a_0_t)
        track_a = math.radians(track_a)
        v_ENz_a_0_t = np.array(
            [
                ground_speed_a * math.sin(track_a),
                ground_speed_a * math.cos(track_a),
                vertical_rate_a,
            ]
        )
        r_ENz_a_1_t = r_ENz_a_0_t + v_ENz_a_0_t * self.lead_time

        # Compute position, at time one, and velocity, at time zero,
        # in the geocentric (XYZ) coordinate system of the aircraft
        # relative to the tripod
        r_XYZ_a_1_t = np.matmul(self.E_XYZ_to_ENz.transpose(), r_ENz_a_1_t)
        v_XYZ_a_0_t = np.matmul(self.E_XYZ_to_ENz.transpose(), v_ENz_a_0_t)

        # Compute the distance between the aircraft and the tripod at
        # time one
        # TODO: Restore?
        # distance3d = ptz_utilities.norm(r_ENz_a_1_t)

        # Compute the distance between the aircraft and the tripod
        # along the surface of a spherical Earth
        # TODO: Restore?
        # distance2d = ptz_utilities.compute_great_circle_distance(
        #     self.lambda_t,
        #     self.varphi_t,
        #     lambda_a,
        #     varphi_a,
        # )  # [m]

        # Compute the bearing from north of the aircraft from the
        # tripod
        # TODO: Restore?
        # bearing = math.degrees(math.atan2(r_ENz_a_1_t[0], r_ENz_a_1_t[1]))

        # Compute pan and tilt to point the camera at the aircraft
        r_uvw_a_1_t = np.matmul(self.E_XYZ_to_uvw, r_XYZ_a_1_t)
        self.rho_a = math.degrees(math.atan2(r_uvw_a_1_t[0], r_uvw_a_1_t[1]))  # [deg]
        self.tau_a = math.degrees(
            math.atan2(r_uvw_a_1_t[2], ptz_utilities.norm(r_uvw_a_1_t[0:2]))
        )  # [deg]

        # Get camera pan, tilt, and zoom
        if self.use_camera:
            self.rho_a, self.tau_a, self.zoom = self.camera_control.get_ptz()

        # Compute slew rate differences
        self.delta_rho_dot_c = self.pan_gain * (self.rho_a - self.rho_c)
        self.delta_tau_dot_c = self.tilt_gain * (self.tau_a - self.tau_c)

        # Compute position and velocity in the camera fixed (rst)
        # coordinate system of the aircraft relative to the tripod at
        # time zero after pointing the camera at the aircraft
        _, _, _, _, _, _, self.E_XYZ_to_rst = ptz_utilities.compute_camera_rotations(
            self.e_E_XYZ,
            self.e_N_XYZ,
            self.e_z_XYZ,
            self.alpha,
            self.beta,
            self.gamma,
            self.rho_a,
            self.tau_a,
        )
        self.r_rst_a_0_t = np.matmul(self.E_XYZ_to_rst, r_XYZ_a_0_t)
        self.v_rst_a_0_t = np.matmul(self.E_XYZ_to_rst, v_XYZ_a_0_t)

        # Compute aircraft slew rate
        omega = (
            ptz_utilities.cross(self.r_rst_a_0_t, self.v_rst_a_0_t)
            / ptz_utilities.norm(self.r_rst_a_0_t) ** 2
        )
        self.rho_dot_a = math.degrees(-omega[2])
        self.tau_dot_a = math.degrees(omega[0])

        # Update camera pan and tilt rate
        self.rho_dot_c = self.rho_dot_a + self.delta_rho_dot_c
        self.tau_dot_c = self.tau_dot_a + self.delta_tau_dot_c

        # Command camera rates, and begin capturing images
        if self.use_camera:
            logger.info(
                f"Commanding pan and tilt rates: {self.rho_dot_c}, {self.tau_dot_c} [deg/s]"
            )
            self.camera_control.continuous_move(
                self._get_pan_rate(self.rho_dot_c),
                self._get_tilt_rate(self.tau_dot_c),
                self.zoom,
            )
            logger.info(f"Starting image capture of aircraft: {self.icao24}")
            self.do_capture = True

        # Log camera pointing using MQTT
        if self.log_to_mqtt:
            msg = {
                "timestamp": str(int(datetime.utcnow().timestamp())),
                "data": {
                    "camera-pointing": {
                        "time_c": self.time_c,
                        "rho_a": self.rho_a,
                        "tau_a": self.tau_a,
                        "rho_dot_a": self.rho_dot_a,
                        "tau_dot_a": self.tau_dot_a,
                        "rho_c": self.rho_c,
                        "tau_c": self.tau_c,
                        "rho_dot_c": self.rho_dot_c,
                        "tau_dot_c": self.tau_dot_c,
                    }
                },
            }
            logger.info(f"Publishing logger msg: {msg}")
            self.publish_to_topic(self.logger_topic, json.dumps(msg))

    def _get_pan_rate(self, rho_dot):
        """Compute pan rate index between -100 and 100 using rates in
        deg/s, limiting the results to the specified minimum and
        maximum.

        Parameters
        ----------
        rho_dot : float
            Pan rate [deg/s]

        Returns
        -------
        pan_rate : int
            Pan rate index
        """
        if rho_dot < self.pan_rate_min:
            pan_rate = -100

        elif self.pan_rate_max < rho_dot:
            pan_rate = +100

        else:
            pan_rate = (
                200
                / (self.pan_rate_max - self.pan_rate_min)
                * (rho_dot - self.pan_rate_min)
                - 100
            )
        return pan_rate

    def _get_tilt_rate(self, tau_dot):
        """Compute tilt rate index between -100 and 100 using rates in
        deg/s, limiting the results to the specified minimum and
        maximum.

        Parameters
        ----------
        tau_dot : float
            Tilt rate [deg/s]

        Returns
        -------
        tilt_rate : int
            Tilt rate index
        """
        if tau_dot < self.tilt_rate_min:
            tilt_rate = -100

        elif self.tilt_rate_max < tau_dot:
            tilt_rate = +100

        else:
            tilt_rate = (
                200
                / (self.tilt_rate_max - self.tilt_rate_min)
                * (tau_dot - self.tilt_rate_min)
                - 100
            )
        return tilt_rate

    def _capture_image(self):
        """Capture a JPEG image, noting the time, if enabled.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.do_capture:
            self.capture_time = time()
            logger.info(
                f"Capturing image of aircraft: {self.icao24}, at: {self.capture_time}"
            )
            # TODO: Implement context manager to move into required capture directory?
            self.camera_configuration.get_jpeg_request(
                camera=1,
                resolution=self.jpeg_resolution,
                compression=self.jpeg_compression,
            )

    def _update_pointing(self):
        """Update values of camera pan and tilt using current pan and
        tilt rate. Note that these value likely differ slightly from
        the actual camera pan and tilt angles, and will be overwritten
        when processing a flight message. The values are used for
        development and testing.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.time_c += self.update_interval
        self.rho_c += self.rho_dot_c * self.update_interval
        self.tau_c += self.tau_dot_c * self.update_interval

    def main(self: Any) -> None:
        """Schedule module heartbeat and image capture, subscribe to
        all required topics, then loop forever. Update pointing for
        logging, and stop capturing images after twice the capture
        interval has elapsed.
        """
        if self.use_mqtt:

            # Schedule module heartbeat
            heartbeat_job = schedule.every(self.heartbeat_interval).seconds.do(
                self.publish_heartbeat, payload="PTZ Controller Module Heartbeat"
            )

            # Subscribe to required topics
            self.add_subscribe_topic(self.config_topic, self._config_callback)
            self.add_subscribe_topic(self.calibration_topic, self._calibration_callback)
            self.add_subscribe_topic(self.flight_topic, self._flight_callback)

        if self.use_camera:

            # Schedule image capture
            capture_job = schedule.every(self.capture_interval).seconds.do(
                self._capture_image, payload=""
            )

        # Enter the main loop
        while True:
            try:
                # Run pending scheduled messages
                if self.use_mqtt:
                    schedule.run_pending()

                # Update camera pointing
                sleep(self.update_interval)
                if not self.use_camera:
                    self._update_pointing()

                # Stop capturing images if a flight message has not
                # been received in twice the capture interval
                if (
                    self.do_capture
                    and time() - self.capture_time > 2.0 * self.capture_interval
                ):
                    logger.info(f"Stopping image capture of aircraft: {self.icao24}")
                    self.do_capture = False

            except Exception as e:
                self.logger.error(f"Main loop exception: {e}")


if __name__ == "__main__":
    # Instantiate controller and execute
    ptz_controller = PtzController(
        camera_ip=os.environ.get("CAMERA_IP"),
        camera_user=os.environ.get("CAMERA_USER"),
        camera_password=os.environ.get("CAMERA_PASSWORD"),
        mqtt_ip=os.environ.get("MQTT_IP"),
        config_topic=os.environ.get("CONFIG_TOPIC"),
        calibration_topic=os.environ.get("CALIBRATION_TOPIC"),
        flight_topic=os.environ.get("FLIGHT_TOPIC"),
        logger_topic=os.environ.get("LOGGER_TOPIC"),
        heartbeat_interval=float(os.environ.get("HEARTBEAT_INTERVAL")),
        update_interval=float(os.environ.get("UPDATE_INTERVAL")),
        capture_interval=float(os.environ.get("CAPTURE_INTERVAL")),
        lead_time=float(os.environ.get("LEAD_TIME")),
        pan_gain=float(os.environ.get("PAN_GAIN")),
        pan_rate_min=float(os.environ.get("PAN_RATE_MIN")),
        pan_rate_max=float(os.environ.get("PAN_RATE_MAX")),
        tilt_gain=float(os.environ.get("TILT_GAIN")),
        tilt_rate_min=float(os.environ.get("TILT_RATE_MIN")),
        tilt_rate_max=float(os.environ.get("TILT_RATE_MAX")),
        jpeg_resolution=os.environ.get("JPEG_RESOLUTION"),
        jpeg_compression=os.environ.get("JPEG_COMPRESSION"),
        use_mqtt=strtobool(os.environ.get("USE_MQTT")),
        use_camera=strtobool(os.environ.get("USE_CAMERA")),
        log_to_mqtt=strtobool(os.environ.get("LOG_TO_MQTT")),
    )
    ptz_controller.main()
