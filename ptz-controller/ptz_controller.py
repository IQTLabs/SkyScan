from datetime import datetime
import json
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

# TODO: Agree on a method for importing the base class
sys.path.append(str(Path(os.getenv("CORE_PATH")).expanduser()))
from base_mqtt_pub_sub import BaseMQTTPubSub
import ptz_utilities

# TODO: Add logging


class PtzController(BaseMQTTPubSub):
    """TODO: Complete"""

    def __init__(
        self: Any,
        config_topic: str,
        calibration_topic: str,
        flight_topic: str,
        lead_time: float,
        update_interval: float,
        gain_pan: float,
        gain_tilt: float,
        debug: bool = False,
        **kwargs: Any,
    ):
        """Initialize the PTZ controller.

        Parameters
        ----------
        config_topic: str
            MQTT topic for subscribing to configuration messages
        calibration_topic: str
            MQTT topic for subscribing to calibration messages
        flight_topic: str
            MQTT topic for subscribing to flight messages
        lead_time: float
            Lead time when assigning pointing to the aircraft [s]
        update_interval: float
            Update interval to compute pointing of the camera [s]
        gain_pan: float
            Proportional control gain for pan
        gain_tilt: float
            Proportional control gain for tilt
        debug: bool
            Flag to debug the PTZ controller, or not

        Returns
        -------
        PtzController
        """
        # Parent class handles kwargs
        super().__init__(**kwargs)
        # self.env_variable = env_variable
        self.config_topic = config_topic
        self.calibration_topic = calibration_topic
        self.flight_topic = flight_topic
        self.lead_time = lead_time
        self.update_interval = update_interval
        self.gain_pan = gain_pan
        self.gain_tilt = gain_tilt
        self.debug = debug

        # Connect client on construction
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

        # Yaw, pitch, and roll angles
        self.alpha = 0.0  # [deg]
        self.beta = 0.0  # [deg]
        self.gamma = 0.0  # [deg]

        # Yaw, pitch, and roll rotation quaternions
        self.q_alpha = None
        self.q_beta = None
        self.q_gamma = None

        # Orthogonal transformation matrix from geocentric to camera
        # housing fixed coordinates
        self.E_XYZ_to_uvw = None

        # Aircraft pan and tilt angles
        self.rho_a = 0.0  # [deg]
        self.tau_a = 0.0  # [deg]

        # Pan, and tilt rotation quaternions
        self.q_row = None
        self.q_tau = None

        # Orthogonal transformation matrix from camera housing to
        # camer fixed coordinates
        self.E_XYZ_to_rst = None

        # Aircraft pan and tilt rates
        self.rho_dot_a = 0.0  # [deg/s]
        self.tau_dot_a = 0.0  # [deg/s]

        # Camera pan and tilt angles
        self.rho_c = 0.0  # [deg]
        self.tau_c = 0.0  # [deg]

        # Camera pan and tilt rates
        self.rho_dot_c = 0.0  # [deg/s]
        self.tau_dot_c = 0.0  # [deg/s]

        # Camera pan and tilt rate differences
        self.delta_rho_dot_c = 0.0  # [deg/s]
        self.delta_tau_dot_c = 0.0  # [deg/s]

        # Time of flight message
        self.time_a = 0.0  # [s]
        # Position of aircraft
        self.r_rst_a_0_t = None  # [m/s]
        # Position of velocity
        self.v_rst_a_0_t = None  # [m/s]

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
        if self.debug:
            payload = msg["data"]
        else:
            payload = self.decode_payload(msg)
        time_a = payload["latLonTime"]  # [sec]
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

        # Compute slew rate differences
        self.time_a = time_a
        self.delta_rho_dot_c = self.gain_pan * (self.rho_a - self.rho_c)
        self.delta_tau_dot_c = self.gain_tilt * (self.tau_a - self.tau_c)

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
        print("hi")

    def _update_pointing(self):
        """Update camera pointing at the update interval.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Assuming constant aircraft velocity and update aircraft
        # position
        self.r_rst_a_0_t += self.v_rst_a_0_t * self.update_interval

        # Compute camera slew rate
        omega = (
            ptz_utilities.cross(self.r_rst_a_0_t, self.v_rst_a_0_t)
            / ptz_utilities.norm(self.r_rst_a_0_t) ** 2
        )
        self.rho_dot_c = math.degrees(-omega[2]) + self.delta_rho_dot_c
        self.tau_dot_c = math.degrees(omega[0]) + self.delta_tau_dot_c

        # Update camera pan and tilt
        dt_c = self.update_interval
        self.rho_c += self.rho_dot_c * dt_c
        self.tau_c += self.tau_dot_c * dt_c

        # TODO: Publish pointing
        # example_data = {
        #     "timestamp": str(int(datetime.utcnow().timestamp())),
        #     "data": "Example data payload",
        # }
        # self.publish_to_topic(self.example_publish_topic, json.dumps(example_data))

    def main(self: Any) -> None:
        """TODO: Complete"""
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
                schedule.run_pending()
                self._update_pointing()
                sleep(self.update_interval)

            except Exception as e:
                if self.debug:
                    print(e)


if __name__ == "__main__":
    # Instantiate controller and execute
    ptz_controller = PtzController(
        config_topic=os.environ.get("CONFIG_TOPIC"),
        calibration_topic=os.environ.get("CALIBRATION_TOPIC"),
        flight_topic=os.environ.get("FLIGHT_TOPIC"),
        lead_time=float(os.environ.get("LEAD_TIME")),
        update_interval=float(os.environ.get("UPDATE_INTERVAL")),
        gain_pan=float(os.environ.get("GAIN_PAN")),
        gain_tilt=float(os.environ.get("GAIN_TILT")),
        mqtt_ip=os.environ.get("MQTT_IP"),
    )
    ptz_controller.main()
