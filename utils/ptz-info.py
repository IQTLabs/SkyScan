import urllib.parse
import datetime
import requests
from bs4 import BeautifulSoup
from requests.auth import HTTPDigestAuth

class CameraConfiguration:
    """
    Module for configuration cameras AXIS
    """

    def __init__(self, ip, user, password):
        self.cam_ip = ip
        self.cam_user = user
        self.cam_password = password

    def get_info(self):  # 5.1.4
        """
        Reload factory default. All parameters are set to their factory default value.
        Returns:
            Success (OK) or Failure (error and description).
        """
        url = 'http://' + self.cam_ip + '/axis-cgi/com/ptzconfig.cgi?info=1'
        resp = requests.get(url, auth=HTTPDigestAuth(self.cam_user, self.cam_password))

        if resp.status_code == 200:
            return resp.text

        text = str(resp)
        text += str(resp.text)
        return text


ip = '11.111.1111.111111'
login = 'xxxx'
password = 'PLACEHOLDER_FOR_PASSWORD'

cam = CameraConfiguration(ip, login, password)
print(cam.get_info())