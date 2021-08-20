# Configure Camera

## Axis Security Camera

- Locate your camera on your network using the [Axis Discovery Tool](https://www.axis.com/support/downloads/axis-ip-utility) or with the following linux command `avahi-browse -a -r`
- Add IP and login info to `.env` file 
> AXIS_USERNAME= # The username for the Axis camera
> AXIS_PASSWORD= # The Password for the Axis camera
> AXIS_IP= # The IP address of the camera

## (Optional) Enable Raspi-camera

If you are using the Pan Tilt hat, you will need to make sure the Pi Camera has been configured correctly:
In the base OS on the Pi make sure the Camera is enabled:
```bash
sudo raspi-config
```
- Interfacing Options
- Camera
- Enable

```bash
echo 'SUBSYSTEM=="vchiq",MODE="0666"' > /etc/udev/rules.d/99-camera.rules
```