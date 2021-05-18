# Design Notes and Decisions

## Units of Measure

- All distances will be measured in **Meters**
- All angles will be measured in **Degrees**
- Coordinates will be in **Decimal Degrees**


## MQTT Topics and Message Formats


### Topic: "skyscan/heartbeat"
*Publised by: egi, tracker, axis-ptz containers*

This topic is used for clients to periodically publish a string message on the MQTT bus. This helps make sure the connection stays up and does not time out. All MQTT clients should do this.

---

### Topic: "skyscan/config/json"
*Published by: Jupyter Notebook*
*Subscribed by: tracker, axis-ptz containers*

The JSON blob has different config values. There are no required fields. The following Keys are used:
- **cameraZoom** - int value from 0 - 9999
- **cameraDelay** - float value for the number of seconds to wait after sending the camera move API command, before sending the take picture API command.
- **cameraMoveSpeed** - This is how fast the camea will move. It is an Int number between 0-99 (max)
- **cameraLead - This is how far the tracker should move the center point in front of the currently tracked plane. It is a float, and is measured in seconds, example: 0.25 . It is based on the planes current heading and how fast it is going. 
- **cameraBearing** - This is a float to correct the cameras heading to help it better align with True North. It can be from -180 to 180. 
- **minElevation** - The minimum elevation above the horizon which the Tracker will follow an airplane. Int value between 0-90 degrees.

---

### Topic: "skyscan/egi"
*Published by: egi container*
*Subscribed by: tracker container*


The JSON blob contains current position information, as reported by the GPS unit. Information on how level the camera is also published. This data comes from an IMU unit attached to the camera. If the GPS or IMU is not available, default information from the command line is sent instead.
- **time** - the time from the GPS, in string format: %Y-%m-%dT%H:%M:%SZ
- **lat** - latitude from the GPS as a float in decimal degrees
- **long** - longitude from the GPS as a float in decimal degrees
- **alt** - altitude from the GPS as an int in meters
- **roll** - roll of the camera platform in degrees
- **pitch** - pitch of the camera platform in degrees
- **yaw** - yaw of the camera platform in degrees
- **fix** - the number of satellites the GPS receiver has acquired

---

### Topic: "skyscan/flight/json"
*Published by: tracker container*
*Subscribed by: axis-ptz container*


The JSON blob contains information about the current plane being tracked, as well as how to position the camera to photograph it. The rate of publication varies based on the distance of the aircraft from the camera.

- **time** - the current time, form the Pi
- **verticalRate** - the vertical rate of climb for the aircraft
- **lat** - the latitude of the aircraft, in decimal degrees
- **lon** - the longitude of the aircraft, in decimal degrees
- **altitude** - the altitude of the aircraft, in meters
- **groundSpeed** - the ground speed of the aircraft, in knots
- **icao24** - the ICAO24 identifier of the aircraft
- **registration** - the FAA registration identifier for the aircraft. This is only available if the aircraft was found in the DB.
- **track** - the heading of the aircraft, in degreees
- **operator** - the operator of the aircraft. This is only available if the aircraft was found in the DB.
- **manufacturer** the manufacturer of the aircraft. This is only available if the aircraft was found in the DB.
- **model** the model of the aircraft. This is only available if the aircraft was found in the DB.
- **callsign** an eight digit flight ID - can be flight number or registration (or even nothing).
- **bearing** the perspective bearing of the aircraft being tracked. It is in degrees, with 0 being the nose of the aircraft, preceding clockwise around the aircraft.
- **elevation** the perspective elevation of the aircraft, in degrees.
- **distance** the distance of the aircraft from the camera, calculated in 3D space, in meters
- **cameraPan** the pan setting for the PTZ camera to point at the aircraft. Any corrections need to correct for the camera being unlevel will be included in this value. It is in degrees.
- **cameraTilt** the tilt setting for the PTZ camera to point at the aircraft. Any corrections need to correct for the camera being unlevel will be included in this value. It is in degrees.


        