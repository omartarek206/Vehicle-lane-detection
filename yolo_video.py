import argparse
import glob
import time
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s-%(name)s-%(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="",
                    help="path to input video file")
parser.add_argument("-o", "--output", type=str, default="",
                    help="path to (optional) output video file")
parser.add_argument("-d", "--display", type=int, default=1,
                    help="display output or not (1/0)")
parser.add_argument("-ht", "--height", type=int, default=1200,
                    help="height of output")
parser.add_argument("-wt", "--width", type=int, default=700,
                    help="width of output")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="confidence threshold")
parser.add_argument("-t", "--threshold", type=float, default=0.4,
                    help="non-maximum supression threshold")
parser.add_argument("-y", "--yolo", type=str, default="yolov4-tiny",
                    help="path to yolo directory")
                   
args = parser.parse_args()
logger.info("Parsed Arguments")

CONFIDENCE_THRESHOLD = args.confidence
NMS_THRESHOLD = args.threshold
if not Path(args.input).exists():
    raise FileNotFoundError("Path to video file is not exist.")

vc = cv2.VideoCapture(args.input)
fps=vc.get(cv2.CAP_PROP_FPS)
if args.yolo=="yolov4-tiny":
    weights ="yolo/yolov4-tiny.weights"
    labels ="yolo/labels.txt"
    cfg ="yolo/yolov4-tiny.cfg"
elif args.yolo=="yolov4":
    weights ="yolov4/yolov4.weights"
    labels ="yolov4/labels.txt"
    cfg ="yolov4/yolov4.cfg"


logger.info("Using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))

class_names = list()
with open(labels, "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer = net.getLayerNames()
layer = [layer[i - 1] for i in net.getUnconnectedOutLayers()]
writer = None

