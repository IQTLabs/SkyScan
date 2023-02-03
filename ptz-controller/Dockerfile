FROM iqtlabs/edgetech-core:latest
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
ADD ./ptz_controller.py .
ADD ./ptz_utilities.py .
ENTRYPOINT [ "python3", "ptz_controller.py" ]
