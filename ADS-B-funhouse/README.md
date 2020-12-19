**ADS-B funhouse**
==========

This is a collection of Python scripts for playing with [ADS-B](https://en.wikipedia.org/wiki/Automatic_dependent_surveillance-broadcast) data from [dump1090](https://github.com/antirez/dump1090). You will need an [rtl-sdr](http://sdr.osmocom.org/trac/wiki/rtl-sdr) receiver to join the fun.

## flighttracker.py

This script reads SBS1 messages from your dump1090 receiver and tracks the nearest aircraft. It publishes information about the flight on MQTT once every second with an increasing frequency if the aircraft is closer.

Install the requirements:

`% sudo -H pip3 install -r requirements.txt`

and start the tracker:

`% ./flighttracker.py  -H <dump1090 host> -m <MQTT host> -pdb <plane database host> -l <receiver latitude> -L <reciver longitude> --prox <MQTT proximity topic>`

The published message is a string of JSON data containing the following fields:

| Key          |  Description                              | Sample data                   |
| ------------ | ----------------------------------------- | ----------------------------- |
| icao24       | ICAO24 designator                         | "4787B0"
| distance     | Distance to aircraft [km]                 | 42
| bearing      | Bearing of aircraft [degrees]             | 42
| icao24       | ICAO24 designator                         | "4787B0"
| loggedDate   | Local timestamp                           | "2015-09-08 21:08:26.061000" 
| operator     | Name of airline                           | "Cathay Pacific Airways"
| type         | Type of aircraft                          | "Boeing 777 367ER"
| registration | Aircrafts ICAO registration               | "B-KPY"
| callsign     | Flight's callsign                         | "CPA257"
| heading      | Aircraf's heading [degrees]               | 131 
| groundSpeed  | Ground speed [knots]                      | 413 
| altitude     | Altitude [feet]                           | 17500 
| image        | Image URL                                 | https://...
| lon          | Lontitude                                 | 13.33108 
| lat          | Latitude                                  | 55.29126
| vspeed.      | Vertical climb/descend rate [ft/min]      | 2240


### Some notes

The aircraft's operator, type and registration are not available in the ADS-B data the aircraft transmits and needs to be pulled from another data source. These are hard to come by as no public database exists that allows robots, to my knowledge. You will need to do some scraping.

The script is designed to utilize a "Plane database server" you need to host by running

`% ./planedb-serv.py flightdata.sql 31541`

Starting the script will create an empty sqlite database for you to polulate with whatever scraped data you can find (ico24 -> aircraft type, registration, and operator).

When all is set, clone my [skygrazer git](https://github.com/kanflo/adsb-skygrazer) to have your Raspberry Pi display the data produced by flighttracker.

This git previously contained adsbclient.py and proxclient.py, both have been deprecated.


## airline-colors.py (will be deprecated)

This script allows commercial pilots to, unknowingly I might add, change your moodlight. Any MQTT controllable moodlight can be set to light up in the prominent color of the airline's logo, dimmed accodring to distance to the plane.

Subscribing to the JSON data from `proxclient.py`, it fetches the logo for the airline that operates the nearest flight and calculates the prominent color of their logo. The color is dimmed according to distance and posted to an MQTT topic.

The prominent color in the logo is the one found in the most pixels, white and black excluded. Colors are cached in a file called `logocolors.json`.

`% airline-colors.py -m <MQTT host> -d <max distance> -t <color topic>`

The default publish topic is `airlinecolor` containing the message `#RRGGBB`

The following arguments are supported by:

| Key         | Description                                                       |
| ------------| ---------------------------------------------------- |
| --help      | well...
| --mqtt-host | MQTT broker hostname
| --mqtt-port | MQTT broker port number (default 1883)
| --distance  | max distance in kilometers, the color will be black (#000000) for aircrafts beyond this distance
| --topic     | the topic to post color data to
| --verbose   | Verbose output

-
Released under the MIT license. Have fun!
