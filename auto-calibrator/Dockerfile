FROM iqtlabs/edgetech-core:latest
RUN apt install build-essential -y
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
ADD ./auto_calibrator.py .
ADD ./ptz_utilities.py .
ENTRYPOINT [ "python3", "auto_calibrator.py" ]
