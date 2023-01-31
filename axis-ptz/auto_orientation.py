import os
from time import sleep
import json
from typing import Any, Dict
import schedule
from datetime import datetime
import paho.mqtt.client as mqtt
from scipy.optimize import fmin_bfgs
import math
import numpy as np

import utils
from edgetech_core.base_mqtt_pub_sub import BaseMQTTPubSub


# inherit functionality from BaseMQTTPubSub parent this way
class TemplatePubSub(BaseMQTTPubSub):
    def __init__(
        self: Any,
        env_variable: Any,
        example_topic: str,
        debug: bool = False,
        **kwargs: Any,
    ):
        # Pass enviornment variables as parameters (include **kwargs) in super().__init__()
        super().__init__(**kwargs)
        self.env_variable = env_variable
        self.yolo_correction_topic = example_topic
        # include debug version
        self.debug = debug

        # Connect client in constructor
        self.connect_client()
        sleep(1)
        self.publish_registration("Template Module Registration")

    def _calculate_pointing_error(
        self: Any, _client: mqtt.Client, _userdata: Dict[Any, Any], msg: Any
    ) -> None:
        # Decode message:
        # Always publishing a JSON string with {timestamp: ____, data: ____,}
        # TODO: more on this to come w/ a JSON header after talking to Rob
        msg = json.loads(str(msg.payload.decode("utf-8")))

        # Do something when a message is recieved
        rho_0, tau_0, rho_epslion, tau_epsilon = self.calculate_pointing_error(msg)

        self.minimize(msg, rho_0, tau_0, rho_epslion, tau_epsilon)

        pass

    @staticmethod
    def calculate_pointing_error(msg):

        # Get values at time of message
        rho_0 = msg["pan"]
        tau_0 = msg["tilt"]

        # Use YOLO or equiv to find pointing error

        # Throw error if camera not at max zoom (9999)
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
    def p_epsilon(alpha_beta_gamma,  # Independent vars
    msg, rho_0, tau_0, rho_epsilon, tau_epsilon):  # Parameters

        # Compute position of the aircraft
        a_varphi = msg["lat"]  # [deg]
        a_lambda = msg["lon"]  # [deg]
        a_h = msg["altitude"]  # [m]
        r_XYZ_a = utils.compute_r_XYZ(a_lambda, a_varphi, a_h)

        # Compute position of the tripod
        t_varphi = msg["camera"]["lat"]  # [deg]
        t_lambda = msg["camera"]["lon"]  # [deg]
        t_h = msg["camera"]["lon"]  # [m]
        r_XYZ_t = utils.compute_r_XYZ(t_lambda, t_varphi, t_h)

        # Compute orthogonal transformation matrix from geocentric to
        # topocentric coordinates, and corresponding unit vectors
        # system of the tripod
        E_XYZ_to_ENz, e_E_XYZ, e_N_XYZ, e_z_XYZ = utils.compute_E(t_lambda, t_varphi)

        # Compute the rotations from the XYZ coordinate system to the uvw
        # (camera housing fixed) coordinate system
        alpha = alpha_beta_gamma[0]  # [deg]
        beta = alpha_beta_gamma[1]  # [deg]
        gamma = alpha_beta_gamma[2]  # [deg]
        _, _, _, E_XYZ_to_uvw, _, _, _ = utils.compute_rotations(
            e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, 0.0, 0.0
        )

        # Compute position in the uvw coordinate system of the aircraft
        # relative to the tripod
        r_uvw_a_t = np.matmul(E_XYZ_to_uvw, r_XYZ_a - r_XYZ_t)

        # Compute pan and tilt to point the camera at the aircraft given
        # the updated values of alpha, beta, and gamma
        rho = math.degrees(math.atan2(r_uvw_a_t[0], r_uvw_a_t[1]))  # [deg]
        tau = math.degrees(
            math.atan2(r_uvw_a_t[2], utils.norm(r_uvw_a_t[0:2]))
        )  # [deg]

        # Return the pointing error to be minimized
        return math.sqrt((rho_0 + rho_epsilon - rho) ** 2 + (tau_0 + tau_epsilon - tau) ** 2)

    def minimize(self, msg, rho_0, tau_0, rho_epsilon, tau_epsilon):

        # Initial guess
        x0 = [0, 0, 0]

        fmin_bfgs(self.p_epsilon, x0, args=[msg, rho_0, tau_0, rho_epsilon, tau_epsilon])

    def main(self: Any) -> None:
        # main funciton wraps functionality and always includes a while True
        # (make sure to include a sleep)

        # include schedule heartbeat in every main()
        schedule.every(10).seconds.do(
            self.publish_heartbeat, payload="Template Module Heartbeat"
        )

        # If subscribing to a topic:
        self.add_subscribe_topic(self.yolo_correction_topic, self._calculate_pointing_error)

        example_data = {
            "timestamp": str(int(datetime.utcnow().timestamp())),
            "data": "Example data payload",
        }

        # example publish data every 10 minutes
        schedule.every(10).minutes.do(
            self.publish_to_topic,
            topic_name="Template Module Heartbeat",
            publish_payload=json.dumps(example_data),
        )

        # or just publish once
        self.publish_to_topic(self.example_publish_topic, json.dumps(example_data))

        while True:
            try:
                # run heartbeat and anything else scheduled if clock is up
                schedule.run_pending()
                # include a sleep so loop does not run at CPU time
                sleep(0.001)

            except Exception as e:
                if self.debug:
                    print(e)


if __name__ == "__main__":
    # instantiate an instance of the class
    # any variables in BaseMQTTPubSub can be overriden using **kwargs
    # and enviornment variables should be in the docker compose (in a .env file)
    template = TemplatePubSub(
        env_variable=os.environ.get("ENV_VARIABLE"),
        yolo_correction_topic=str(os.environ.get("YOLO_CORRECTION_TOPIC")),
        mqtt_ip=os.environ.get("MQTT_IP"),
    )
    # call the main function
    template.main()