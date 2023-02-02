FROM iqtlabs/edgetech-core:latest
COPY requirements.txt .
RUN pip3 install -r requirements.txt
ADD ./template_pub_sub.py .
ENTRYPOINT [ "python3", "template_pub_sub.py" ]