#! /bin/bash 
jupyter notebook \
--notebook-dir=/opt/notebooks \
--ip='*' \
--port=8888 \
--no-browser \
--allow-root \
--NotebookApp.password="$(cat /run/secrets/login_info)"