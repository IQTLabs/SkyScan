#!/usr/bin/env bash

# commands to compile protocol buffer description files and
# to install packages necessary for tensorflow.

# note: it is potentially preferable to handle the python
# package installation by including these packages in the
# requirements.txt, but John Speed M. on 7/2/2021 decided
# with Luke B. that following Google's lead, who originally
# created this code in a demo Jupyter notebook is preferable
# to ensure correctness.

cd /tf/models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .