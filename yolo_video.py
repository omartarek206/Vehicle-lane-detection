#importing necessary libraries
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
#defining parser arguments
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
#threshold of drawing boxes in the output
CONFIDENCE_THRESHOLD = args.confidence
#threshold for handling overlapped boxes by iou then non maximum suppression
NMS_THRESHOLD = args.threshold
#if user doesnot enter path of input video file
if not Path(args.input).exists():
    raise FileNotFoundError("Path to video file is not exist.")
#reading video as an array of frames
vc = cv2.VideoCapture(args.input)
#getting the no of frames per second to pass it to function that writes the output file
fps=vc.get(cv2.CAP_PROP_FPS)

#"yolo-tiny" files path
if args.yolo=="yolov4-tiny":
    weights ="yolo/yolov4-tiny.weights"
    labels ="yolo/labels.txt"
    cfg ="yolo/yolov4-tiny.cfg"
#"yolo" files path
elif args.yolo=="yolov4":
    weights ="yolov4/yolov4.weights"
    labels ="yolov4/labels.txt"
    cfg ="yolov4/yolov4.cfg"

#displaying yolo files to user
logger.info("Using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))

#reading class names
class_names = list()
with open(labels, "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#Randomly generate the colors(the R value, G value and B value) using np.random.randint for each label(thus the size (len(LABELS), 3). These colors will be used to highlight the detected object of that class later.
COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8")

#reading parameters of yolo darknet
net = cv2.dnn.readNetFromDarknet(cfg, weights)
#prefer to use gpu
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#determine only the *output* layer names that we need from YOLO
layer = net.getLayerNames()
layer = [layer[i - 1] for i in net.getUnconnectedOutLayers()]
#initialize the writer
writer = None

#detect function
def detect(frm, net, ln):
    #frame dimensions
    (H, W) = frm.shape[:2]
    #A binary large object (BLOB or blob) is a collection of binary data stored as a single entity to be an input to darknet
    blob = cv2.dnn.blobFromImage(frm, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    #calculate time of each frame
    start_time = time.time()
    #forward propagation of frame in the darknet
    layerOutputs = net.forward(ln)
    end_time = time.time()
    
    #initializing list of boxes,classes and confidences
    boxes = []
    classIds = []
    confidences = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                classIds.append(classID)
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIds[i]]]
            cv2.rectangle(frm, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(class_names[classIds[i]], confidences[i])
            cv2.putText(
                frm, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )

            fps_label = "FPS: %.2f" % (1 / (end_time - start_time))
            cv2.putText(
                frm, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )
            

t1=time.time()

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        break
    frame = cv2.resize(frame, (args.height, args.width))
    detect(frame, net, layer)

    # if args.display == 1:
    #     # cv2.imshow("detections", frame)


    if args.output != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            args.output, fourcc, fps, (frame.shape[1], frame.shape[0]), True
        )

    if writer is not None:
        writer.write(frame)
t2=time.time()
print("Time taken : {0} minutes".format((t2-t1)/60))
