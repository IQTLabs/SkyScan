#!/usr/bin/env python3


import argparse
from datetime import datetime, timedelta
import errno
import json
from json.decoder import JSONDecodeError
import math
import os
import random
import requests
import sys
import threading
import time

import logging
import logging.config  # This gets rid of the annoying log messages from Vapix_Control
import coloredlogs

import numpy as np
from requests.auth import HTTPDigestAuth
import paho.mqtt.client as mqtt
from sensecam_control import vapix_control  # , vapix_config

import utils

# Logging configuration
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
logging.getLogger("vapix_control.py").setLevel(logging.WARNING)
logging.getLogger("vapix_control").setLevel(logging.WARNING)
logging.getLogger("sensecam_control").setLevel(logging.WARNING)
root_logger = logging.getLogger()
if not root_logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
logger = logging.getLogger("camera")
logger.setLevel(logging.INFO)

ID = str(random.randint(1, 100001))
args = None
camera = None
cameraBearingCorrection = 0
cameraConfig = None
cameraZoom = None
cameraMoveSpeed = None
cameraDelay = None
cameraLead = 0
inhibitPhotos = False
capturePeriod = 1000  # milliseconds
active = False
Active = True

object_topic = None
flight_topic = None
config_topic = "skyscan/config/json"

bearing = 0  # this is an angle
elevation = 0  # this is an angle
cameraPan = 0  # This value is in angles
cameraTilt = 0  # This values is in angles
distance3d = 0  # this is in Meters
distance2d = 0  # in meters
angularVelocityHorizontal = 0  # in meters
angularVelocityVertical = 0  # in meters
planeTrack = 0  # This is the direction that the plane is moving in

currentPlane = None

camera_latitude = None
camera_longitude = None
camera_altitude = None
camera_lead = None


def calculate_bearing_correction(b):
    return (b + cameraBearingCorrection) % 360


# Copied from VaPix/Sensecam to customize the folder structure for saving pictures
def get_jpeg_request():  # 5.2.4.1
    """
    The requests specified in the JPEG/MJPG section are supported by those video products
    that use JPEG and MJPG encoding.
    Args:
        resolution: Resolution of the returned image. Check the product’s Release notes.
        camera: Selects the source camera or the quad stream.
        square_pixel: Enable/disable square pixel correction. Applies only to video encoders.
        compression: Adjusts the compression level of the image.
        clock: Shows/hides the time stamp. (0 = hide, 1 = show)
        date: Shows/hides the date. (0 = hide, 1 = show)
        text: Shows/hides the text. (0 = hide, 1 = show)
        text_string: The text shown in the image, the string must be URL encoded.
        text_color: The color of the text shown in the image. (black, white)
        text_background_color: The color of the text background shown in the image.
        (black, white, transparent, semitransparent)
        rotation: Rotate the image clockwise.
        text_position: The position of the string shown in the image. (top, bottom)
        overlay_image: Enable/disable overlay image.(0 = disable, 1 = enable)
        overlay_position:The x and y coordinates defining the position of the overlay image.
        (<int>x<int>)
    Returns:
        Success ('image save' and save the image in the file folder) or Failure (Error and
        description).
    """
    payload = {
        "resolution": "1920x1080",
        "compression": 5,
        "camera": 1,
    }
    global args

    url = "http://" + args.axis_ip + "/axis-cgi/jpg/image.cgi"
    start_time = datetime.now()
    try:
        resp = requests.get(
            url,
            auth=HTTPDigestAuth(args.axis_username, args.axis_password),
            params=payload,
            timeout=0.5,
        )
    except requests.exceptions.Timeout:
        logging.info("🚨 Images capture request timed out 🚨  ")
        return

    disk_time = datetime.now()
    if resp.status_code == 200:
        captureDir = None

        if args.flat_file_structure:
            captureDir = "capture/"
        else:
            captureDir = "capture/{}".format(currentPlane["type"])
        try:
            os.makedirs(captureDir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..
        filename = "{}/{}_{}_{}_{}_{}.jpg".format(
            captureDir,
            currentPlane["icao24"],
            int(bearing),
            int(elevation),
            int(distance3d),
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        )

        # Original
        with open(filename, "wb") as var:
            var.write(resp.content)

        # Non-Blocking
        # fd = os.open(filename, os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK)
        # os.write(fd, resp.content)
        # os.close(fd)

        # Blocking

        # fd = os.open(filename, os.O_CREAT | os.O_WRONLY)
        # os.write(fd, resp.content)
        # os.close(fd)
    else:
        logging.error(
            "Unable to fetch image: {}\tstatus: {}".format(url, resp.status_code)
        )

    end_time = datetime.now()
    net_time_diff = disk_time - start_time
    disk_time_diff = end_time - disk_time
    if disk_time_diff.total_seconds() > 0.1:
        logging.info(
            "🚨  Image Capture Timeout  🚨  Net time: {}  \tDisk time: {}".format(
                net_time_diff, disk_time_diff
            )
        )


def get_bmp_request():  # 5.2.4.1
    """
    The requests specified in the JPEG/MJPG section are supported by those video products
    that use JPEG and MJPG encoding.
    Args:
        resolution: Resolution of the returned image. Check the product’s Release notes.
        camera: Selects the source camera or the quad stream.
        square_pixel: Enable/disable square pixel correction. Applies only to video encoders.
        compression: Adjusts the compression level of the image.
        clock: Shows/hides the time stamp. (0 = hide, 1 = show)
        date: Shows/hides the date. (0 = hide, 1 = show)
        text: Shows/hides the text. (0 = hide, 1 = show)
        text_string: The text shown in the image, the string must be URL encoded.
        text_color: The color of the text shown in the image. (black, white)
        text_background_color: The color of the text background shown in the image.
        (black, white, transparent, semitransparent)
        rotation: Rotate the image clockwise.
        text_position: The position of the string shown in the image. (top, bottom)
        overlay_image: Enable/disable overlay image.(0 = disable, 1 = enable)
        overlay_position:The x and y coordinates defining the position of the overlay image.
        (<int>x<int>)
    Returns:
        Success ('image save' and save the image in the file folder) or Failure (Error and
        description).
    """
    payload = {
        "resolution": "1920x1080",
        "camera": 1,
    }
    url = "http://" + args.axis_ip + "/axis-cgi/bitmap/image.bmp"
    resp = requests.get(
        url, auth=HTTPDigestAuth(args.axis_username, args.axis_password), params=payload
    )

    if resp.status_code == 200:
        captureDir = "capture/{}".format(currentPlane["type"])
        try:
            os.makedirs(captureDir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..
        filename = "{}/{}_{}_{}_{}_{}.bmp".format(
            captureDir,
            currentPlane["icao24"],
            int(bearing),
            int(elevation),
            int(distance3d),
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        )

        with open(filename, "wb") as var:
            var.write(resp.content)
        return str("Image saved")

    text = str(resp)
    text += str(resp.text)
    return text




def calculateCameraPositionB(
    r_XYZ_t, E_XYZ_to_ENz, e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, E_XYZ_to_uvw
):
    """Calculates camera pointing at a specified lead time."""
    # Define global variables
    # TODO: Eliminate use of global variables
    global distance3d
    global distance2d
    global bearing
    global elevation
    global angularVelocityHorizontal
    global angularVelocityVertical
    global cameraPan
    global cameraTilt

    # Assign position and velocity of the aircraft
    a_varphi = currentPlane["lat"]  # [deg]
    a_lambda = currentPlane["lon"]  # [deg]
    # currentPlane["latLonTime"]
    a_h = currentPlane["altitude"]  # [m]
    # currentPlane["altitudeTime"]
    a_track = currentPlane["track"]  # [deg]
    a_ground_speed = currentPlane["groundSpeed"]  # [m/s]
    a_vertical_rate = currentPlane["verticalRate"]  # [m/s]
    # currentPlane["icao24"]
    # currentPlane["type"]

    # Assign position of the tripod
    t_varphi = camera_latitude  # [deg]
    t_lambda = camera_longitude  # [deg]

    # Compute position in the XYZ coordinate system of the aircraft
    # relative to the tripod at time zero, the observation time
    r_XYZ_a_0 = utils.compute_r_XYZ(a_lambda, a_varphi, a_h)
    r_XYZ_a_0_t = r_XYZ_a_0 - r_XYZ_t

    # Compute position and velocity in the ENz coordinate system of
    # the aircraft relative to the tripod at time zero, and position at
    # slightly later time one
    r_ENz_a_0_t = np.matmul(E_XYZ_to_ENz, r_XYZ_a_0 - r_XYZ_t)
    a_track = math.radians(a_track)
    v_ENz_a_0_t = np.array(
        [
            a_ground_speed * math.sin(a_track),
            a_ground_speed * math.cos(a_track),
            a_vertical_rate,
        ]
    )
    r_ENz_a_1_t = r_ENz_a_0_t + v_ENz_a_0_t * camera_lead

    # Compute position, at time one, and velocity, at time zero, in
    # the XYZ coordinate system of the aircraft relative to the tripod
    r_XYZ_a_1_t = np.matmul(E_XYZ_to_ENz.transpose(), r_ENz_a_1_t)
    v_XYZ_a_0_t = np.matmul(E_XYZ_to_ENz.transpose(), v_ENz_a_0_t)

    # Compute the distance between the aircraft and the tripod at time
    # one
    distance3d = utils.norm(r_ENz_a_1_t)

    # Compute the distance between the aircraft and the tripod along
    # the surface of a spherical Earth
    distance2d = utils.compute_great_circle_distance(
        t_varphi,
        t_lambda,
        a_varphi,
        a_lambda,
    )  # [m]

    # Compute the bearing from north of the aircraft from the tripod
    bearing = math.degrees(math.atan2(r_ENz_a_1_t[0], r_ENz_a_1_t[1]))

    # Compute pan and tilt to point the camera at the aircraft
    r_uvw_a_1_t = np.matmul(E_XYZ_to_uvw, r_XYZ_a_1_t)
    rho = math.degrees(math.atan2(r_uvw_a_1_t[0], r_uvw_a_1_t[1]))  # [deg]
    tau = math.degrees(
        math.atan2(r_uvw_a_1_t[2], utils.norm(r_uvw_a_1_t[0:2]))
    )  # [deg]
    cameraPan = rho
    cameraTilt = tau

    # Compute position and velocity in the rst coordinate system of
    # the aircraft relative to the tripod at time zero after pointing
    # the camera at the aircraft
    _, _, _, _, q_rho, q_tau, E_XYZ_to_rst = utils.compute_rotations(
        e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, rho, tau
    )
    r_rst_a_0_t = np.matmul(E_XYZ_to_rst, r_XYZ_a_0_t)
    v_rst_a_0_t = np.matmul(E_XYZ_to_rst, v_XYZ_a_0_t)

    # Compute the components of the angular velocity of the aircraft
    # in the rst coordinate system
    omega = utils.cross(r_rst_a_0_t, v_rst_a_0_t) / utils.norm(r_rst_a_0_t) ** 2
    angularVelocityHorizontal = math.degrees(-omega[2])
    angularVelocityVertical = math.degrees(omega[0])


def calculateCameraPositionA():
    global cameraPan
    global cameraTilt
    global distance2d
    global distance3d
    global bearing
    global angularVelocityHorizontal
    global angularVelocityVertical
    global elevation

    (lat, lon, alt) = utils.calc_travel_3d(currentPlane, camera_lead)
    distance3d = utils.coordinate_distance_3d(
        camera_latitude, camera_longitude, camera_altitude, lat, lon, alt
    )
    # (latorig, lonorig) = utils.calc_travel(observation.getLat(), observation.getLon(), observation.getLatLonTime(),  observation.getGroundSpeed(), observation.getTrack(), camera_lead)
    distance2d = utils.coordinate_distance(camera_latitude, camera_longitude, lat, lon)
    bearing = utils.bearingFromCoordinate(
        cameraPosition=[camera_latitude, camera_longitude],
        airplanePosition=[lat, lon],
        heading=currentPlane["track"],
    )
    elevation = utils.elevation(
        distance2d, cameraAltitude=camera_altitude, airplaneAltitude=alt
    )
    (angularVelocityHorizontal, angularVelocityVertical) = utils.angular_velocity(
        currentPlane, camera_latitude, camera_longitude, camera_altitude
    )
    # logging.info("Angular Velocity - Horizontal: {} Vertical: {}".format(angularVelocityHorizontal, angularVelocityVertical))
    cameraTilt = elevation
    cameraPan = utils.cameraPanFromCoordinate(
        cameraPosition=[camera_latitude, camera_longitude], airplanePosition=[lat, lon]
    )
    cameraPan = calculate_bearing_correction(cameraPan)


def moveCamera(ip, username, password):

    movePeriod = 100  # milliseconds
    moveTimeout = datetime.now()
    captureTimeout = datetime.now()
    camera = vapix_control.CameraControl(ip, username, password)

    # Assign position of the tripod
    t_varphi = camera_latitude  # [deg]
    t_lambda = camera_longitude  # [deg]
    t_h = camera_altitude  # [m]

    # Compute orthogonal transformation matrix from geocentric to
    # topocentric coordinates, and position in the XYZ coordinate
    # system of the tripod
    E_XYZ_to_ENz, e_E_XYZ, e_N_XYZ, e_z_XYZ = utils.compute_E(t_lambda, t_varphi)
    r_XYZ_t = utils.compute_r_XYZ(t_lambda, t_varphi, t_h)

    # Compute the rotations from the XYZ coordinate system to the uvw
    # (camera housing fixed) coordinate system
    alpha = 0.0  # [deg]
    beta = 0.0  # [deg]
    gamma = 0.0  # [deg]
    q_alpha, q_beta, q_gamma, E_XYZ_to_uvw, _, _, _ = utils.compute_rotations(
        e_E_XYZ, e_N_XYZ, e_z_XYZ, alpha, beta, gamma, 0.0, 0.0
    )

    while True:
        if active:
            if not "icao24" in currentPlane:
                logging.info(" 🚨 Active but Current Plane is not set")
                continue
            if moveTimeout <= datetime.now():
                calculateCameraPositionB(
                    r_XYZ_t,
                    E_XYZ_to_ENz,
                    e_E_XYZ,
                    e_N_XYZ,
                    e_z_XYZ,
                    alpha,
                    beta,
                    gamma,
                    E_XYZ_to_uvw,
                )
                camera.absolute_move(cameraPan, cameraTilt, cameraZoom, cameraMoveSpeed)
                # logging.info("Moving to Pan: {} Tilt: {}".format(cameraPan, cameraTilt))
                moveTimeout = moveTimeout + timedelta(milliseconds=movePeriod)
                if moveTimeout <= datetime.now():
                    lag = datetime.now() - moveTimeout
                    logging.info(
                        " 🚨 Move execution time was greater that Move Period - lag: {}".format(
                            lag
                        )
                    )
                    moveTimeout = datetime.now() + timedelta(milliseconds=movePeriod)

            if not inhibitPhotos:
                if captureTimeout <= datetime.now():
                    time.sleep(cameraDelay)
                    get_jpeg_request()
                    captureTimeout = captureTimeout + timedelta(
                        milliseconds=capturePeriod
                    )
                    if captureTimeout <= datetime.now():
                        lag = datetime.now() - captureTimeout
                        logging.info(
                            " 🚨 Capture execution time was greater that Capture Period - lag: {}".format(
                                lag
                            )
                        )
                        captureTimeout = datetime.now() + timedelta(
                            milliseconds=capturePeriod
                        )
            delay = 0.005
            time.sleep(delay)
        else:
            delay = 1
            time.sleep(delay)


def update_config(config):
    global cameraZoom
    global cameraMoveSpeed
    global cameraDelay
    global cameraPan
    global camera_lead
    global camera_altitude
    global cameraBearingCorrection
    global inhibitPhotos
    global capturePeriod

    if "cameraZoom" in config:
        cameraZoom = int(config["cameraZoom"])
        logging.info("Setting Camera Zoom to: {}".format(cameraZoom))
    if "cameraDelay" in config:
        cameraDelay = float(config["cameraDelay"])
        logging.info("Setting Camera Delay to: {}".format(cameraDelay))
    if "cameraMoveSpeed" in config:
        cameraMoveSpeed = int(config["cameraMoveSpeed"])
        logging.info("Setting Camera Move Speed to: {}".format(cameraMoveSpeed))
    if "cameraLead" in config:
        camera_lead = float(config["cameraLead"])
        logging.info("Setting Camera Lead to: {}".format(camera_lead))
    if "cameraAltitude" in config:
        camera_altitude = float(config["cameraAltitude"])
        logging.info("Setting Camera Altitude to: {}".format(camera_altitude))
    if "cameraBearingCorrection" in config:
        cameraBearingCorrection = float(config["cameraBearingCorrection"])
        logging.info(
            "Setting Camera Bearing Correction to: {}".format(cameraBearingCorrection)
        )
    if "inhibitPhotos" in config:
        inhibitPhotos = bool(config["inhibitPhotos"])
        if inhibitPhotos:
            logging.info("Setting Camera to inhibit photos")
        else:
            logging.info("Setting Camera to save photos")
    if "capturePeriod" in config:
        capturePeriod = float(config["capturePeriod"])
        logging.info("Setting Camera Capture Period (sec) to: {}".format(capturePeriod))


#############################################
##         MQTT Callback Function          ##
#############################################


def on_message(client, userdata, message):
    try:
        return on_message_impl(client, userdata, message)
    except Exception as exc:
        logging.exception("Error in MQTT message callback: %s", exc)


def on_message_impl(client, userdata, message):
    global currentPlane
    global object_timeout
    global camera_longitude
    global camera_latitude
    global camera_altitude

    global active

    command = str(message.payload.decode("utf-8"))
    # rint(command)
    try:
        update = json.loads(command)
        # payload = json.loads(messsage.payload) # you can use json.loads to convert string to json
    except JSONDecodeError as e:
        # do whatever you want
        logging.exception("Error decoding message as JSON: %s", e)
    except TypeError as e:
        logging.exception("Error decoding message as JSON: %s", e)
    # do whatever you want in this case
    except ValueError as e:
        logging.exception("Error decoding message as JSON: %s", e)
    except Exception as e:
        logging.exception("Error decoding message as JSON: %s", e)
        print("Caught it!")

    if message.topic == object_topic:
        logging.info("Got Object Topic")
        # TODO: Resolve reference
        setXY(update["x"], update["y"])
        object_timeout = time.mktime(time.gmtime()) + 5
    elif message.topic == flight_topic:
        if "icao24" in update:
            if active is False:
                logging.info("{}\t[Starting Capture]".format(update["icao24"]))
            logging.info(
                "{}\t[IMAGE]\tBearing: {} \tElv: {} \tDist: {}".format(
                    update["icao24"],
                    int(update["bearing"]),
                    int(update["elevation"]),
                    int(update["distance"]),
                )
            )
            currentPlane = update
            active = True
        else:
            if active is True:
                logging.info("{}\t[Stopping Capture]".format(currentPlane["icao24"]))
            active = False
            # It is better to just have the old values for currentPlane in case a message comes in while the
            # moveCamera Thread is running.
            # currentPlane = {}
    elif message.topic == config_topic:
        update_config(update)
        logging.info("Config Message: {}".format(update))
    elif message.topic == "skyscan/egi":
        # logging.info(update)
        camera_longitude = float(update["long"])
        camera_latitude = float(update["lat"])
        camera_altitude = float(update["alt"])
    else:
        logging.info(
            "Message: {} Object: {} Flight: {}".format(
                message.topic, object_topic, flight_topic
            )
        )


def on_disconnect(client, userdata, rc):
    global Active
    Active = False
    logging.error("Axis-PTZ MQTT Disconnect!")


def main():
    global args
    global logging
    global camera
    global cameraDelay
    global cameraMoveSpeed
    global cameraZoom
    global cameraPan
    global camera_altitude
    global camera_latitude
    global camera_longitude
    global camera_lead
    global cameraConfig
    global flight_topic
    global object_topic
    global Active

    parser = argparse.ArgumentParser(description="An MQTT based camera controller")
    parser.add_argument("--lat", type=float, help="Latitude of camera")
    parser.add_argument("--lon", type=float, help="Longitude of camera")
    parser.add_argument(
        "--alt", type=float, help="altitude of camera in METERS!", default=0
    )
    parser.add_argument(
        "--camera-lead",
        type=float,
        help="how many seconds ahead of a plane's predicted location should the camera be positioned",
        default=0.1,
    )

    parser.add_argument(
        "-m", "--mqtt-host", help="MQTT broker hostname", default="127.0.0.1"
    )
    parser.add_argument(
        "-t",
        "--mqtt-flight-topic",
        help="MQTT topic to subscribe to",
        default="skyscan/flight/json",
    )
    parser.add_argument(
        "--mqtt-object-topic",
        help="MQTT topic to subscribe to",
        default="skyscan/object/json",
    )
    parser.add_argument(
        "-u", "--axis-username", help="Username for the Axis camera", required=True
    )
    parser.add_argument(
        "-p", "--axis-password", help="Password for the Axis camera", required=True
    )
    parser.add_argument(
        "-a", "--axis-ip", help="IP address for the Axis camera", required=True
    )
    parser.add_argument(
        "-s",
        "--camera-move-speed",
        type=int,
        help="The speed at which the Axis will move for Pan/Tilt (0-100)",
        default=50,
    )
    parser.add_argument(
        "-d",
        "--camera-delay",
        type=float,
        help="How many seconds after issuing a Pan/Tilt command should a picture be taken",
        default=0,
    )
    parser.add_argument(
        "-z",
        "--camera-zoom",
        type=int,
        help="The zoom setting for the camera (0-9999)",
        default=9999,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-f",
        "--flat-file-structure",
        action="store_true",
        help="Use a flat file structure (all images saved to ./) rather than organizing images in folder by plane type.",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO

    styles = {
        "critical": {"bold": True, "color": "red"},
        "debug": {"color": "green"},
        "error": {"color": "red"},
        "info": {"color": "white"},
        "notice": {"color": "magenta"},
        "spam": {"color": "green", "faint": True},
        "success": {"bold": True, "color": "green"},
        "verbose": {"color": "blue"},
        "warning": {"color": "yellow"},
    }
    level = (
        logging.DEBUG if "-v" in sys.argv or "--verbose" in sys.argv else logging.INFO
    )
    if 1:
        coloredlogs.install(
            level=level,
            fmt="%(asctime)s.%(msecs)03d \033[0;90m%(levelname)-8s "
            ""
            "\033[0;36m%(filename)-18s%(lineno)3d\033[00m "
            "%(message)s",
            level_styles=styles,
        )
    else:
        # Show process name
        coloredlogs.install(
            level=level,
            fmt="%(asctime)s.%(msecs)03d \033[0;90m%(levelname)-8s "
            "\033[0;90m[\033[00m \033[0;35m%(processName)-15s\033[00m\033[0;90m]\033[00m "
            "\033[0;36m%(filename)s:%(lineno)d\033[00m "
            "%(message)s",
        )

    logging.info(
        "---[ Starting %s ]---------------------------------------------" % sys.argv[0]
    )
    # camera = vapix_control.CameraControl(args.axis_ip, args.axis_username, args.axis_password)
    cameraDelay = args.camera_delay
    cameraMoveSpeed = args.camera_move_speed
    cameraZoom = args.camera_zoom
    camera_longitude = args.lon
    camera_latitude = args.lat
    camera_altitude = args.alt  # Altitude is in METERS
    camera_lead = args.camera_lead
    # cameraConfig = vapix_config.CameraConfiguration(args.axis_ip, args.axis_username, args.axis_password)

    cameraMove = threading.Thread(
        target=moveCamera,
        args=[args.axis_ip, args.axis_username, args.axis_password],
        daemon=True,
    )
    cameraMove.start()
    # Sleep for a bit so we're not hammering the HAT with updates
    delay = 0.005
    time.sleep(delay)
    flight_topic = args.mqtt_flight_topic
    object_topic = args.mqtt_object_topic
    print(
        "connecting to MQTT broker at "
        + args.mqtt_host
        + ", channel '"
        + flight_topic
        + "'"
    )
    client = mqtt.Client("skyscan-axis-ptz-camera-" + ID)  # create new instance

    client.on_message = on_message  # attach function to callback
    client.on_disconnect = on_disconnect

    client.connect(args.mqtt_host)  # connect to broker
    client.loop_start()  # start the loop
    client.subscribe(flight_topic)
    client.subscribe(object_topic)
    client.subscribe(config_topic)
    client.subscribe("skyscan/egi")
    client.publish(
        "skyscan/registration",
        "skyscan-axis-ptz-camera-" + ID + " Registration",
        0,
        False,
    )

    #############################################
    ##                Main Loop                ##
    #############################################
    timeHeartbeat = 0
    while Active:
        if timeHeartbeat < time.mktime(time.gmtime()):
            timeHeartbeat = time.mktime(time.gmtime()) + 10
            client.publish(
                "skyscan/heartbeat",
                "skyscan-axis-ptz-camera-" + ID + " Heartbeat",
                0,
                False,
            )
            if not cameraMove.is_alive():
                logging.critical(
                    "Thread within Axis-PTZ has failed!  Killing container."
                )
                Active = False

        delay = 0.1
        time.sleep(delay)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        logging.critical(e, exc_info=True)
