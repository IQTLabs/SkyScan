from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
from time import sleep
from typing import Any, Dict
import sys

import numpy as np
import quaternion
import paho.mqtt.client as mqtt
import schedule
from sensecam_control import vapix_control

# TODO: Agree on a method for importing the base class
sys.path.append(str(Path(os.getenv("CORE_PATH")).expanduser()))
from base_mqtt_pub_sub import BaseMQTTPubSub
import ptz_utilities

logger = logging.getLogger("ptz_controller")
logger.setLevel(logging.INFO)


class PtzController(BaseMQTTPubSub):
    """TODO: Complete"""

    def __init__(
        self: Any,
        camera_ip: str,
        camera_user: str,
        camera_password: str,
        config_topic: str,
        calibration_topic: str,
        flight_topic: str,
        heartbeat_interval: float,
        update_interval: float,
        lead_time: float,
        pan_gain: float,
        pan_rate_min: float,
        pan_rate_max: float,
        tilt_gain: float,
        tilt_rate_min: float,
        tilt_rate_max: float,
        debug: bool = False,
        **kwargs: Any,
    ):
        """Instanstiate the PTZ controller by connecting to the
        message broker, and initializing data attributes.

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
        heartbeat_interval: float
            Interval at which heartbeat message is to be published [s]
        update_interval: float
            Interval at which pointing of the camera is computed [s]
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
        debug: bool
            Flag to debug the PTZ controller, or not

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
        self.heartbeat_interval = heartbeat_interval
        self.update_interval = update_interval
        self.lead_time = lead_time
        self.pan_gain = pan_gain
        self.pan_rate_min = pan_rate_min
        self.pan_rate_max = pan_rate_max
        self.tilt_gain = tilt_gain
        self.tilt_rate_min = tilt_rate_min
        self.tilt_rate_max = tilt_rate_max
        self.debug = debug

        # Construct camera control
        self.camera_control = vapix_control.CameraControl(
            self.camera_ip, self.camera_user, self.camera_password
        )

        # Connect MQTT client
        if not self.debug:
            self.connect_client()
            sleep(1)
            self.publish_registration("Template Module Registration")

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

        # Time of flight message and corresponding aircraft position
        # and velocity
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

        # Camera pan and tilt angles, and zoom
        self.rho_c = 0.0  # [deg]
        self.tau_c = 0.0  # [deg]
        self.zoom = 0  # -100 to 100 [-]

        # Camera pan and tilt rates
        self.rho_dot_c = 0.0  # [deg/s]
        self.tau_dot_c = 0.0  # [deg/s]

        # Camera pan and tilt rate differences
        self.delta_rho_dot_c = 0.0  # [deg/s]
        self.delta_tau_dot_c = 0.0  # [deg/s]

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
        # TODO: Complete
        if self.debug:
            payload = msg["data"]
        else:
            payload = self.decode_payload(msg)
        self.lambda_t = payload["tripod_longitude"]  # [deg]
        self.varphi_t = payload["tripod_latitude"]  # [deg]
        self.h_t = payload["tripod_altitude"]  # [m]

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
        # TODO: Complete
        if self.debug:
            payload = msg["data"]
        else:
            payload = self.decode_payload(msg)
        self.alpha = payload["tripod_yaw"]  # [deg]
        self.beta = payload["tripod_pitch"]  # [deg]
        self.gamma = payload["tripod_roll"]  # [deg]

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
        # TODO: Complete
        if self.debug:
            payload = msg["data"]
        else:
            payload = self.decode_payload(msg)
        self.time_a = payload["latLonTime"]  # [sec]
        lambda_a = payload["lon"]  # [deg]
        varphi_a = payload["lat"]  # [deg]
        h_a = payload["altitude"]  # [m]
        track_a = payload["track"]  # [deg]
        ground_speed_a = payload["groundSpeed"]  # [m/s]
        vertical_rate_a = payload["verticalRate"]  # [m/s]

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
        if not self.debug:
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

        # Command camera rates
        if not self.debug:
            self.camera_control.continuous_move(
                200
                / (self.pan_rate_max - self.pan_rate_min)
                * (self.rho_dot_c - self.pan_rate_min)
                - 100,
                200
                / (self.tilt_rate_max - self.tilt_rate_min)
                * (self.rho_dot_c - self.tilt_rate_min)
                - 100,
                self.zoom,
            )

    def update_pointing(self):
        """Update values of camera pan and tilt using current pan and
        tilt rate. Note that these value likely differ slightly from
        the actual camera pan and tilt angles.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.rho_c += self.rho_dot_c * self.update_interval
        self.tau_c += self.tau_dot_c * self.update_interval

    def main(self: Any) -> None:
        """TODO: Complete"""

        if not self.debug:

            # Schedule module heartbeat
            schedule.every(10).seconds.do(
                self.publish_heartbeat, payload="PTZ Controller Module Heartbeat"
            )

            # Subscribe to required topics
            # TODO: Add a topic to exit gracefully?
            self.add_subscribe_topic(self.config_topic, self._config_callback)
            self.add_subscribe_topic(self.calibration_topic, self._calibration_callback)
            self.add_subscribe_topic(self.flight_topic, self._flight_callback)

        # Run pending scheduled messages
        while True:
            try:
                if not self.debug:
                    schedule.run_pending()
                self.update_pointing
                sleep(self.update_interval)

            except Exception as e:
                if self.debug:
                    print(e)


if __name__ == "__main__":
    # Instantiate controller and execute
    ptz_controller = PtzController(
        camera_ip=os.environ.get("CAMERA_IP"),
        camera_user=os.environ.get("CAMERA_USER"),
        camera_password=os.environ.get("CAMERA_PASSWORD"),
        config_topic=os.environ.get("CONFIG_TOPIC"),
        mqtt_ip=os.environ.get("MQTT_IP"),
        calibration_topic=os.environ.get("CALIBRATION_TOPIC"),
        flight_topic=os.environ.get("FLIGHT_TOPIC"),
        heartbeat_interval=float(os.environ.get("HEARTBEAT_INTERVAL")),
        update_interval=float(os.environ.get("UPDATE_INTERVAL")),
        lead_time=float(os.environ.get("LEAD_TIME")),
        pan_gain=float(os.environ.get("PAN_GAIN")),
        pan_rate_min=float(os.environ.get("PAN_RATE_MIN")),
        pan_rate_max=float(os.environ.get("PAN_RATE_MAX")),
        tilt_gain=float(os.environ.get("TILT_GAIN")),
        tilt_rate_min=float(os.environ.get("TILT_RATE_MIN")),
        tilt_rate_max=float(os.environ.get("TILT_RATE_MAX")),
    )
    ptz_controller.main()
