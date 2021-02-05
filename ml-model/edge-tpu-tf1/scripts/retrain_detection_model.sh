#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Starts retraining detection model.

  --num_training_steps Number of training steps to run, 500 by default.
  --num_eval_steps     Number of evaluation steps to run, 100 by default.
  --help               Display this help.
END_OF_USAGE
}

num_training_steps=500
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_training_steps)
      num_training_steps=$2
      shift 2 ;;
    --num_eval_steps)
      num_eval_steps=$2
      shift 2 ;;    
    --help)
      usage
      exit 0 ;;
    --*)
      echo "Unknown flag $1"
      usage
      exit 1 ;;
  esac
done

source "$PWD/constants.sh"

mkdir -p "${TRAIN_DIR}"

python object_detection/model_main.py \
  --pipeline_config_path="${CKPT_DIR}/pipeline.config" \
  --model_dir="${TRAIN_DIR}" \
  --num_train_steps="${num_training_steps}" \
  --num_eval_steps="${num_eval_steps}"
