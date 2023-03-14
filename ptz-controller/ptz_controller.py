from datetime import datetime
from distutils.util import strtobool
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from time import sleep, time
from typing import Any, Dict

import numpy as np
import quaternion
import paho.mqtt.client as mqtt
import schedule
from sensecam_control import vapix_config, vapix_control

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
    controller, and capture images while in track."""

    def __init__(
        self: Any,
        camera_ip: str,
        camera_user: str,
        camera_password: str,
        config_topic: str,
        calibration_topic: str,
        flight_topic: str,
        capture_topic: str,
        logger_topic: str,
        heartbeat_interval: float,
        lambda_t: float = 0.0,
        varphi_t: float = 0.0,
        h_t: float = 0.0,
        update_interval: float = 0.1,
        capture_interval: float = 2.0,
        capture_dir: str = ".",
        lead_time: float = 0.25,
        pan_gain: float = 0.2,
        pan_rate_min: float = 1.8,
        pan_rate_max: float = 150.0,
        tilt_gain: float = 0.2,
        tilt_rate_min: float = 1.8,
        tilt_rate_max: float = 150.0,
        jpeg_resolution: str = "1920x1080",
        jpeg_compression: int = 5,
        use_mqtt: bool = True,
        use_camera: bool = True,
        include_age: bool = True,
        log_to_mqtt: bool = False,
        **kwargs: Any,
    ):
        """Instantiate the PTZ controller by connecting to the camera
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
        capture_topic: str
            MQTT topic for publising capture messages
        logger_topic: str
            MQTT topic for publishing logger messages
        heartbeat_interval: float
            Interval at which heartbeat message is to be published [s]
        lambda_t: float
            Tripod geodetic longitude [deg]
        varphi_t: float = 0.0,
            Tripod geodetic latitude [deg]
        h_t: float = 0.0,
            Tripod geodetic altitude [deg]
        update_interval: float
            Interval at which pointing of the camera is computed [s]
        capture_interval: float
            Interval at which the camera image is captured [s]
        capture_dir: str
            Directory in which to place captured images
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
        include_age: bool
            Flag to include flight message age in lead time, or not
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
        self.capture_topic = capture_topic
        self.logger_topic = logger_topic
        self.heartbeat_interval = heartbeat_interval
        self.lambda_t = lambda_t
        self.varphi_t = varphi_t
        self.h_t = h_t
        self.update_interval = update_interval
        self.capture_interval = capture_interval
        self.capture_dir = capture_dir
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
        self.include_age = include_age
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

        # Aircraft identifier, time of flight message and
        # corresponding aircraft longitude, latitude, and altitude,
        # and position and velocity relative to the tripod in the
        # camera fixed (rst) coordinate system
        self.icao24 = "NA"
        self.time_a = 0.0  # [s]
        self.lambda_a = 0.0  # [deg]
        self.varphi_a = 0.0  # [deg]
        self.h_a = 0.0  # [m]
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

        # Distance between the aircraft and the tripod at time one
        distance3d = 0.0  # [m]

        # Aircraft azimuth and elevation angles
        self.azm_a = 0.0  # [deg]
        self.elv_a = 0.0  # [deg]

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
        self.zoom = 2000  # 1 to 9999 [-]

        # Camera pan and tilt rates
        self.rho_dot_c = 0.0  # [deg/s]
        self.tau_dot_c = 0.0  # [deg/s]

        # Camera pan and tilt rate differences
        self.delta_rho_dot_c = 0.0  # [deg/s]
        self.delta_tau_dot_c = 0.0  # [deg/s]

        # Capture boolean and last capture time
        self.do_capture = False
        self.capture_time = 0.0

        # Initialize tripod position in the geocentric (XYZ)
        # coordinate system, orthogonal transformation matrix from
        # geocentric (XYZ) to topocentric (ENz) coordinates, and East,
        # North, and zenith unit vectors
        config_msg = {
            "data": {
                "lambda_t": self.lambda_t,
                "varphi_t": self.varphi_t,
                "h_t": self.h_t,
            }
        }
        self._config_callback(None, None, config_msg)

        # Initialize the rotations from the geocentric (XYZ)
        # coordinate system to the camera housing fixed (uvw)
        # coordinate system
        calibration_msg = {
            "data": {
                "camera": {
                    "tripod_yaw": self.alpha,
                    "tripod_pitch": self.beta,
                    "tripod_roll": self.gamma,
                }
            }
        }
        self._calibration_callback(None, None, calibration_msg)

        # Initialize camera pointing
        if self.use_camera:
            self.camera_control.absolute_move(self.rho_c, self.tau_c, self.zoom, 100)

        # TODO: Log configuration parameters
        logger.info(
            f"""PtzController initialized with parameters:

    camera_ip = {camera_ip}
    camera_user = {camera_user}
    camera_password = {camera_password}
    config_topic = {config_topic}
    calibration_topic = {calibration_topic}
    flight_topic = {flight_topic}
    capture_topic = {capture_topic}
    logger_topic = {logger_topic}
    heartbeat_interval = {heartbeat_interval}
    lambda_t = {lambda_t}
    varphi_t = {varphi_t}
    h_t = {h_t}
    update_interval = {update_interval}
    capture_interval = {capture_interval}
    capture_dir = {capture_dir}
    lead_time = {lead_time}
    pan_gain = {pan_gain}
    pan_rate_min = {pan_rate_min}
    pan_rate_max = {pan_rate_max}
    tilt_gain = {tilt_gain}
    tilt_rate_min = {tilt_rate_min}
    tilt_rate_max = {tilt_rate_max}
    jpeg_resolution = {jpeg_resolution}
    jpeg_compression = {jpeg_compression}
    use_mqtt = {use_mqtt}
    use_camera = {use_camera}
    include_age = {include_age}
    log_to_mqtt = {log_to_mqtt}
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

    def _config_callback(
        self: Any,
        _client: mqtt.Client,
        _userdata: Dict[Any, Any],
        msg: Any,
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
        # Assign data attributes allowed to change during operation
        if type(msg) == mqtt.MQTTMessage:
            data = self.decode_payload(msg.payload)
        else:
            data = msg["data"]
        logger.info(f"Processing config msg data: {data}")
        self.lambda_t = data.get("tripod_longitude", self.lambda_t)  # [deg]
        self.varphi_t = data.get("tripod_latitude", self.varphi_t)  # [deg]
        self.h_t = data.get("tripod_altitude", self.h_t)  # [m]
        self.update_interval = data.get("update_interval", self.update_interval)  # [s]
        self.capture_interval = data.get(
            "capture_interval", self.capture_interval
        )  # [s]
        self.capture_dir = data.get("capture_dir", self.capture_dir)
        self.lead_time = data.get("lead_time", self.lead_time)  # [s]
        self.pan_gain = data.get("pan_gain", self.pan_gain)  # [1/s]
        self.tilt_gain = data.get("tilt_gain", self.tilt_gain)  # [1/s]
        self.include_age = data.get("include_age", self.include_age)
        self.log_to_mqtt = data.get("log_to_mqtt", self.log_to_mqtt)

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
        if type(msg) == mqtt.MQTTMessage:
            data = self.decode_payload(msg.payload)
        else:
            data = msg["data"]
        logger.info(f"Processing calibration msg data: {data}")
        camera = data.get("camera", {})
        self.alpha = camera.get("tripod_yaw", self.alpha)  # [deg]
        self.beta = camera.get("tripod_pitch", self.beta)  # [deg]
        self.gamma = camera.get("tripod_roll", self.gamma)  # [deg]

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
        # Assign identifier, time, position, and velocity of the
        # aircraft
        if type(msg) == mqtt.MQTTMessage:
            data = self.decode_payload(msg.payload)
        else:
            data = msg["data"]
        self.icao24 = data.get("icao24", self.icao24)
        if not set(
            [
                "latLonTime",
                "lon",
                "lat",
                "altitude",
                "track",
                "groundSpeed",
                "verticalRate",
            ]
        ) <= set(data.keys()):
            logger.info(f"Required keys missing from flight message data: {data}")
            return
        logger.info(f"Processing flight msg data: {data}")
        self.time_a = data["latLonTime"]  # [s]
        self.time_c = self.time_a
        self.lambda_a = data["lon"]  # [deg]
        self.varphi_a = data["lat"]  # [deg]
        self.h_a = data["altitude"]  # [m]
        track_a = data["track"]  # [deg]
        ground_speed_a = data["groundSpeed"]  # [m/s]
        vertical_rate_a = data["verticalRate"]  # [m/s]

        # Compute position in the geocentric (XYZ) coordinate system
        # of the aircraft relative to the tripod at time zero, the
        # observation time
        r_XYZ_a_0 = ptz_utilities.compute_r_XYZ(self.lambda_a, self.varphi_a, self.h_a)
        r_XYZ_a_0_t = r_XYZ_a_0 - self.r_XYZ_t

        # Assign lead time, computing and adding age of flight
        # message, if enabled
        lead_time = self.lead_time  # [s]
        if self.include_age:
            datetime_a = ptz_utilities.convert_time(self.time_a)
            lead_time += (datetime.utcnow() - datetime_a).total_seconds()  # [s]
        logger.info(f"Using lead time: {lead_time} [s]")

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
        r_ENz_a_1_t = r_ENz_a_0_t + v_ENz_a_0_t * lead_time

        # Compute position, at time one, and velocity, at time zero,
        # in the geocentric (XYZ) coordinate system of the aircraft
        # relative to the tripod
        r_XYZ_a_1_t = np.matmul(self.E_XYZ_to_ENz.transpose(), r_ENz_a_1_t)
        v_XYZ_a_0_t = np.matmul(self.E_XYZ_to_ENz.transpose(), v_ENz_a_0_t)

        # Compute the distance between the aircraft and the tripod at
        # time one
        distance3d = ptz_utilities.norm(r_ENz_a_1_t)

        # Compute the distance between the aircraft and the tripod
        # along the surface of a spherical Earth
        # TODO: Restore?
        # distance2d = ptz_utilities.compute_great_circle_distance(
        #     self.self.lambda_t,
        #     self.varphi_t,
        #     self.lambda_a,
        #     self.varphi_a,
        # )  # [m]

        # Compute the aircraft azimuth and elevation relative to the
        # tripod
        self.azm_a = math.degrees(math.atan2(r_ENz_a_1_t[0], r_ENz_a_1_t[1]))  # [deg]
        self.elv_a = math.degrees(
            math.atan2(r_ENz_a_1_t[2], ptz_utilities.norm(r_ENz_a_1_t[0:2]))
        )  # [deg]
        logger.info(f"Aircraft azimuth and elevation: {self.azm_a}, {self.elv_a} [deg]")

        # Compute pan and tilt to point the camera at the aircraft
        r_uvw_a_1_t = np.matmul(self.E_XYZ_to_uvw, r_XYZ_a_1_t)
        self.rho_a = math.degrees(math.atan2(r_uvw_a_1_t[0], r_uvw_a_1_t[1]))  # [deg]
        self.tau_a = math.degrees(
            math.atan2(r_uvw_a_1_t[2], ptz_utilities.norm(r_uvw_a_1_t[0:2]))
        )  # [deg]
        logger.info(f"Aircraft pan and tilt: {self.rho_a}, {self.tau_a} [deg]")

        # TODO: Remove
        # if self.rho_a < 30.0 or 100.0 < self.rho_a:
        #     logger.info("Skipping aircraft with pan not in the interval (30, 100)")
        #     self.camera_control.stop_move()
        #     return

        # TODO: Remove
        # Point at the aircraft if the identifier has changed
        # if self.icao24 != icao24:
        #     self.icao24 = icao24
        #     logger.info(f"Pointing at aircraft: {self.icao24}")
        #     self.camera_control.absolute_move(self.rho_a, self.tau_a, self.zoom, 100)
        #     return

        # Get camera pan, tilt, and zoom
        if self.use_camera:
            self.rho_c, self.tau_c, _zoom = self.camera_control.get_ptz()
            logger.info(f"Camera pan and tilt: {self.rho_c}, {self.tau_c} [deg]")
        else:
            logger.info(f"Controller pan and tilt: {self.rho_c}, {self.tau_c} [deg]")

        # Compute slew rate differences
        self.delta_rho_dot_c = self.pan_gain * (self.rho_a - self.rho_c)
        self.delta_tau_dot_c = self.tilt_gain * (self.tau_a - self.tau_c)
        logger.info(
            f"Delta pan and tilt rates: {self.delta_rho_dot_c}, {self.delta_tau_dot_c} [deg/s]"
        )

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
        logger.info(
            f"Aircraft pan and tilt rates: {self.rho_dot_a}, {self.tau_dot_a} [deg/s]"
        )

        # Update camera pan and tilt rate
        self.rho_dot_c = self.rho_dot_a + self.delta_rho_dot_c
        self.tau_dot_c = self.tau_dot_a + self.delta_tau_dot_c
        logger.info(
            f"Camera pan and tilt rates: {self.rho_dot_c}, {self.tau_dot_c} [deg/s]"
        )

        # Command camera rates, and begin capturing images
        if self.use_camera:
            pan_rate_index = self._compute_pan_rate_index(self.rho_dot_c)
            tilt_rate_index = self._compute_tilt_rate_index(self.tau_dot_c)
            logger.info(
                f"Commanding pan and tilt rate indexes: {pan_rate_index}, {tilt_rate_index}"
            )
            self.camera_control.continuous_move(
                pan_rate_index,
                tilt_rate_index,
                0.0,
            )
            if not self.do_capture:
                logger.info(f"Starting image capture of aircraft: {self.icao24}")
                self.do_capture = True
                self.capture_time = time()

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

    def _compute_pan_rate_index(self, rho_dot):
        """Compute pan rate index between -100 and 100 using rates in
        deg/s, limiting the results to the specified minimum and
        maximum. Note that the dead zone from -1.8 to 1.8 deg/s is ignored.

        Parameters
        ----------
        rho_dot : float
            Pan rate [deg/s]

        Returns
        -------
        pan_rate : int
            Pan rate index
        """
        if rho_dot < -self.pan_rate_max:
            pan_rate = -100

        elif self.pan_rate_max < rho_dot:
            pan_rate = +100

        else:
            pan_rate = round((100 / self.pan_rate_max) * rho_dot)
        return pan_rate

    def _compute_tilt_rate_index(self, tau_dot):
        """Compute tilt rate index between -100 and 100 using rates in
        deg/s, limiting the results to the specified minimum and
        maximum. Note that the dead zone from -1.8 to 1.8 deg/s is ignored.

        Parameters
        ----------
        tau_dot : float
            Tilt rate [deg/s]

        Returns
        -------
        tilt_rate : int
            Tilt rate index
        """
        if tau_dot < -self.tilt_rate_max:
            tilt_rate = -100

        elif self.tilt_rate_max < tau_dot:
            tilt_rate = +100

        else:
            tilt_rate = round((100 / self.tilt_rate_max) * tau_dot)
        return tilt_rate

    def _capture_image(self):
        """When enabled, capture an image in JPEG format, and publish
        corresponding image metadata.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.do_capture:

            # Capture an image in JPEG format
            self.capture_time = time()
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            image_filepath = Path(self.capture_dir) / "{}_{}_{}_{}_{}.jpg".format(
                self.icao24,
                int(self.azimuth),
                int(self.elevation),
                int(self.distance3d),
                timestamp,
            )
            logger.info(
                f"Capturing image of aircraft: {self.icao24}, at: {self.capture_time}, in: {self.image_filepath}"
            )
            with tempfile.TemporaryDirectory() as d:
                self.camera_configuration.get_jpeg_request(
                    camera=1,
                    resolution=self.jpeg_resolution,
                    compression=self.jpeg_compression,
                )
                d.glob("*.jpg")[0].rename(image_filepath)

            # Populate and publish image metadata
            image_metadata = {
                "timestamp": timestamp,
                "imagefile": str(image_filepath),
                "camera": {
                    "bearing": self.azimuth,
                    "zoom": self.zoom,
                    "pan": self.rho_c,
                    "tilt": self.tau_c,
                    "lat": self.varphi_t,
                    "long": self.lambda_t,
                    "alt": self.h_t,
                },
                "aircraft": {
                    "lat": self.varphi_a,
                    "long": self.lambda_a,
                    "alt": self.h_a,
                },
            }
            mqtt_client.publish(
                self.capture_topic, json.dumps(image_metadata), 0, False
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
        logging, and command zero camera pan and tilt rates and stop
        capturing images after twice the capture interval has elapsed.
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
                self._capture_image
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

                # Command zero camera pan and tilt rates, and stop
                # capturing images if a flight message has not been
                # received in twice the capture interval
                if (
                    self.do_capture
                    and time() - self.capture_time > 2.0 * self.capture_interval
                ):
                    logger.info("Stopping continuous pan and tilt")
                    self.camera_control.stop_move()
                    logger.info(f"Stopping image capture of aircraft: {self.icao24}")
                    self.do_capture = False

            except Exception as e:
                logger.error(f"Main loop exception: {e}")


def make_controller():
    return PtzController(
        camera_ip=os.getenv("CAMERA_IP"),
        camera_user=os.getenv("CAMERA_USER"),
        camera_password=os.getenv("CAMERA_PASSWORD"),
        mqtt_ip=os.getenv("MQTT_IP"),
        config_topic=os.getenv("CONFIG_TOPIC"),
        calibration_topic=os.getenv("CALIBRATION_TOPIC"),
        flight_topic=os.getenv("FLIGHT_TOPIC"),
        capture_topic=os.getenv("CAPTURE_TOPIC"),
        logger_topic=os.getenv("LOGGER_TOPIC"),
        heartbeat_interval=float(os.getenv("HEARTBEAT_INTERVAL")),
        lambda_t=float(os.getenv("TRIPOD_LONGITUDE")),
        varphi_t=float(os.getenv("TRIPOD_LATITUDE")),
        h_t=float(os.getenv("TRIPOD_ALTITUDE")),
        update_interval=float(os.getenv("UPDATE_INTERVAL")),
        capture_interval=float(os.getenv("CAPTURE_INTERVAL")),
        capture_dir=os.getenv("CAPTURE_DIR"),
        lead_time=float(os.getenv("LEAD_TIME")),
        pan_gain=float(os.getenv("PAN_GAIN")),
        pan_rate_min=float(os.getenv("PAN_RATE_MIN")),
        pan_rate_max=float(os.getenv("PAN_RATE_MAX")),
        tilt_gain=float(os.getenv("TILT_GAIN")),
        tilt_rate_min=float(os.getenv("TILT_RATE_MIN")),
        tilt_rate_max=float(os.getenv("TILT_RATE_MAX")),
        jpeg_resolution=os.getenv("JPEG_RESOLUTION"),
        jpeg_compression=os.getenv("JPEG_COMPRESSION"),
        use_mqtt=strtobool(os.getenv("USE_MQTT")),
        use_camera=strtobool(os.getenv("USE_CAMERA")),
        include_age=strtobool(os.getenv("INCLUDE_AGE")),
        log_to_mqtt=strtobool(os.getenv("LOG_TO_MQTT")),
    )


if __name__ == "__main__":
    # Instantiate controller and execute
    ptz_controller = make_controller()
    ptz_controller.main()
