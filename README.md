# SkyScan
Automatically photograph planes that fly by!
-----
Follow allowing as we build, [here](https://iqtlabs.github.io/SkyScan/)


### Configure
Copy the **env-example** file to **.env**. Edit the **.env** file to include the correct values.

Update **docker-compose.yml** with the correct ID for your RTL-SDR.

### Operations
Launch the application using docker-compose: 
```bash
docker-compose up
```

### Enable Raspi-camera

In the base OS on the Pi make sure the Camera is enabled:
```bash
sudo raspi-config
```
- Interfacing Options
- Camera
- Enable


