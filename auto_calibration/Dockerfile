FROM iqtlabs/edgetech-core:latest
COPY requirements.txt .
RUN pip3 install -r requirements.txt
ADD ./auto_calibrator.py .
ADD ./calibration_utils.py .
ENTRYPOINT [ "python3", "auto_calibrator.py" ]