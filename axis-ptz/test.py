#!/usr/bin/env python3



import argparse
import logging
import coloredlogs
import sys
import os
import time
from sensecam_control import vapix_control,vapix_config


def main():

    global camera
    global cameraConfig

    parser = argparse.ArgumentParser(description='An MQTT based camera controller')


    parser.add_argument('-u', '--axis-username', help="Username for the Axis camera", required=True)
    parser.add_argument('-p', '--axis-password', help="Password for the Axis camera", required=True)
    parser.add_argument('-a', '--axis-ip', help="IP address for the Axis camera", required=True)

    args = parser.parse_args()
    print("hello")

    styles = {'critical': {'bold': True, 'color': 'red'}, 'debug': {'color': 'green'}, 'error': {'color': 'red'}, 'info': {'color': 'white'}, 'notice': {'color': 'magenta'}, 'spam': {'color': 'green', 'faint': True}, 'success': {'bold': True, 'color': 'green'}, 'verbose': {'color': 'blue'}, 'warning': {'color': 'yellow'}}
    level = logging.DEBUG if '-v' in sys.argv or '--verbose' in sys.argv else logging.INFO
    if 1:
        coloredlogs.install(level=level, fmt='%(asctime)s.%(msecs)03d \033[0;90m%(levelname)-8s '
                            ''
                            '\033[0;36m%(filename)-18s%(lineno)3d\033[00m '
                            '%(message)s',
                            level_styles = styles)
    else:
        # Show process name
        coloredlogs.install(level=level, fmt='%(asctime)s.%(msecs)03d \033[0;90m%(levelname)-8s '
                                '\033[0;90m[\033[00m \033[0;35m%(processName)-15s\033[00m\033[0;90m]\033[00m '
                                '\033[0;36m%(filename)s:%(lineno)d\033[00m '
                                '%(message)s')
    print("hello")
    logging.info("---[ Starting %s ]---------------------------------------------" % sys.argv[0])
    camera = vapix_control.CameraControl(args.axis_ip, args.axis_username, args.axis_password)
    print("hello")
    #############################################
    ##                Main Loop                ##
    #############################################
    while True:
        camera.absolute_move(0, 0, 0, 50)
        time.sleep(5)
        camera.absolute_move(90, 0, 0, 50)
        time.sleep(5)
        camera.absolute_move(180, 0, 0, 50)
        time.sleep(5)
        camera.absolute_move(270, 0, 0, 50)
        time.sleep(5)
        camera.absolute_move(0, 90, 0, 50)
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e, exc_info=True)
