# Setting up a Pi from Scratch

## Burn an image to a MicroSD card

There are 2 paths:
	1. Use the Raspberry Pi Imager:
		- https://www.raspberrypi.org/software/
		- It can download an image and burn it to a MicroSD card
	2. Download an OS image and use Balena Etcher to burn it to a card
		- https://www.raspberrypi.org/software/operating-systems/
        - https://www.balena.io/etcher/
        

## Enable SSH
Before you eject the card, enable SSH. You can do this by adding a file called ssh in the root of the card:
	- touch /Volumes/boot/ssh
    - (If you are on a PC the MicroSD card maybe on another path)

 ## Connecting
- Connect the Pi to ethernet and power it on
- figure out the IP address for the Pi. You can do this by checking your router's dashboard. Another option is using `arp -a`. If you only have one Pi on the network, you can also do: `ping raspberrypi.local`
- If you don't already have SSH Keys, create them: `ssh-keygen`
- Pass the SSH Keys to the Pi: `ssh-copy-id pi@raspberrypi.local`
- now test it out: `ssh pi@raspberrypi.local`

## Create an AP
First, update and upgrade:
````
sudo apt-get update
sudo apt-get upgrade    
````

We will be following the directions from Raspberry Pi on [Creating a Routed AP](https://www.raspberrypi.org/documentation/configuration/wireless/access-point-routed.md). I will copy the steps I used, but refer back to this doc if there are doubts or questions.

### Install the access point and network management software

#### Install hostapd, dnsmasq and netfilter stuff
```bash
sudo apt install -y hostapd
```

```bash
sudo systemctl unmask hostapd
sudo systemctl enable hostapd
```

```bash
sudo DEBIAN_FRONTEND=noninteractive apt install -y dnsmasq netfilter-persistent iptables-persistent
```

#### Create a static IP for WLAN

```bash
sudo nano /etc/dhcpcd.conf
```

And add the following at the very end: 

```bash
interface wlan0
    static ip_address=192.168.4.1/24
    nohook wpa_supplicant
```

#### Enable routing and IP masquerading
```bash
sudo nano /etc/sysctl.d/routed-ap.conf
```
And add the following:

```bash
# https://www.raspberrypi.org/documentation/configuration/wireless/access-point-routed.md
# Enable IPv4 routing
net.ipv4.ip_forward=1
```

#### Add Netfilters rule and save it
```bash
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
sudo netfilter-persistent save
```

#### Configure the DHCP and DNS services for the wireless network

```bash
sudo mv /etc/dnsmasq.conf /etc/dnsmasq.conf.orig
sudo nano /etc/dnsmasq.conf
```

Now add the following:

```bash
interface=wlan0 # Listening interface
dhcp-range=192.168.4.2,192.168.4.20,255.255.255.0,24h
                # Pool of IP addresses served via DHCP
domain=wlan     # Local wireless DNS domain
address=/gw.wlan/192.168.4.1
                # Alias for this router
```

#### Turn on Wifi
`sudo rfkill unblock wlan`

#### Configure the access point software

`sudo nano /etc/hostapd/hostapd.conf`

Now add the following... *Edit the ssid to be unique*

```bash
country_code=US
interface=wlan0
ssid=SkyScanMega
hw_mode=g
channel=7
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
```


#### Run your new wireless access point
`sudo systemctl reboot`

After the pi has rebooted, connect to the AP at the SSID you set. Now ssh to it: `ssh pi@192.168.4.1`

#### Turn off
You now effectively have an open AP to your network... probably best to temporarily turn this off:
`sudo rfkill block wlan`

To renable, run the following:

`sudo rfkill unblock wlan`


## Setup SkyScan

I like to store my work in a **Projects** folder:

`mkdir ~/Projects`

Now go there and download SkyScan:

```bash
cd ~/Projects
git clone https://github.com/IQTLabs/SkyScan.git
```

Configure the .env file:

```bash
cd SkyScan
cp env-example .env
nano .env
```

## Add Docker

Install the following prerequisites:

`sudo apt-get install apt-transport-https ca-certificates software-properties-common -y`

Download and install Docker:

`curl -fsSL get.docker.com -o get-docker.sh && sh get-docker.sh`

Give the ‘pi’ user the ability to run Docker:

`sudo usermod -aG docker pi`

Start the Docker service:

`systemctl start docker.service`

Verify that Docker is installed and running (You may need to log out and log back in so that your group membership is re-evaluated):

`docker info`

Install Docker Compose:

`sudo pip3 install docker-compose`

Logout and log back in again, so the permission changes take affect.

## Build containers

```bash
cd ~/Projects/SkyScan
docker-compose build
```

This could take a long, long time....


## Setup RTL-SDR

The RTL-SDR dongle is used to receive the ADS-B broadcasts. The following command is required:

`sudo nano /etc/modprobe.d/blacklist-rtl2832.conf`

```bash
# Blacklist RTL2832 so docker container piaware can use the device

blacklist rtl2832
blacklist dvb_usb_rtl28xxu
blacklist rtl2832_sdr
```

And then remove the kernel module from memory, in case it already got loaded:

```bash
sudo rmmod rtl2832_sdr
sudo rmmod dvb_usb_rtl28xxu
sudo rmmod rtl2832
```

## Configure GPS
Some GPS devices seem to be sensitive to tty echo settings which prevent getting a GPS fix. Use the following command to disable tty echo:

```bash
stty -F /dev/ttyACM0 -echo -echoe -echok
```

Use the following command to verify the GPS device is working:

```bash
gpscat /dev/ttyACM0 | gpsdecode
```

## Wrap it up...

Now go back to the readme to finish the config. You will need to get a Flight Aware key.