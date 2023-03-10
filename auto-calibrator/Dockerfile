FROM iqtlabs/edgetech-core:latest
RUN apt-get update && apt-get install -y \
  build-essential \
  && rm -rf /var/lib/apt/lists/*
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
COPY ./auto_calibrator.py .
COPY ./ptz_utilities.py .
ENTRYPOINT [ "python3", "auto_calibrator.py" ]
