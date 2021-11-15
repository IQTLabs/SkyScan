
## Start up instance
1. Turn on the B VPN
1. Goto SSO
1. Goto AWS SSO
1. Select AWS ACCount
1. Goto SkyScan Maganeent console
1. Goto EC2
1. Select the instace, select it, Start Instance
1. Click on the instance once it is running
1. Get the Private IP for the instance
1. Now goto the Terminal
1. `ssh -i "pth to .pem file" ubuntu@<pirvate.ip.address>`

## Open Notebooks
1. Open browser
1. Goto: `http://<private.ip.adress>:8888`
1. Open the **Notebooks** folder


## Start up Jupyter notebook
*If you leave the Jupyter notebook container running, it should automatically start up again with the AWS Instance starts. If that doesn't happen for some reason, try the following*

Go into the ml-model directory for skyscan
1. `cd ~/SkyScan-Private/ml-model`
1. `docker-compose up`





