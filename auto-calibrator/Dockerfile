FROM iqtlabs/edgetech-core:latest
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
ADD ./auto_calibrator.py .
ADD ../ptz-controller/ptz_utilities.py .
ENTRYPOINT [ "python3", "auto_calibrator.py" ]
