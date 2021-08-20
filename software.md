# Software Operation

SSH to the RaspberryPi and launch the application using docker-compose: 
```bash
cd ~/Projects/SkyScan
docker-compose up
```

A web interface will be available on **port 8080**. As pictures of planes are captured they will be saved in folders in the **./capture** directory.

## Testing with pytest

To run tests with pytest, run:

```bash
pytest
```