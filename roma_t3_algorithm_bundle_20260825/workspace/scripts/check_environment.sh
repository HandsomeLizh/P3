#!/usr/bin/env bash
set -e
set -o pipefail
set +u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/enter_workspace.sh"

python3 - <<'PY'
import sys

import cv2
import numpy
import rclpy
import torch
from grid_map_msgs.msg import GridMap
import roma_vo
from roma_vo import StereoSLAMv2
from std_srvs.srv import Trigger

print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("RoMa VO:", roma_vo.__file__)
print("StereoSLAMv2:", StereoSLAMv2)
print("GridMap message:", GridMap)
print("Save service:", Trigger)
print("Environment check: OK")
PY

ros2 pkg prefix t1_dataset_sim
ros2 pkg prefix t2_basic_semantics
ros2 pkg prefix t3_interfaces
ros2 pkg prefix t3_semantic_mapping
ros2 pkg prefix grid_map_msgs
ros2 pkg prefix t3_upstream_adapters
