import numpy as np
import matplotlib.image as mpimg
import cv2


from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
import sys


class FindLaneLines:

    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def pipeline(self,im, final, bird_eye, thresholded, thresholded_lane, lane):
        height, width = 1080, 1920
        FinalScreen = np.zeros((height, width, 3), dtype=np.uint8)
        FinalScreen[0:360, 1280:1920] = cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA)
        FinalScreen[360:720, 1280:1920] = cv2.resize(bird_eye, (640, 360), interpolation=cv2.INTER_AREA)
        thresholded_frame_cpy = np.copy(thresholded)
        thresholded_frame_cpy = np.dstack((thresholded_frame_cpy, thresholded_frame_cpy, thresholded_frame_cpy))
        FinalScreen[720:1080, 1280:1920] = cv2.resize(thresholded_frame_cpy, (640, 360), interpolation=cv2.INTER_AREA)
        FinalScreen[720:1080, 0:640] = cv2.resize(lane, (640, 360), interpolation=cv2.INTER_AREA)
        FinalScreen[720:1080, 640:1280] = cv2.resize(thresholded_lane, (640, 360), interpolation=cv2.INTER_AREA)
        FinalScreen[0:720, 0:1280] = cv2.resize(final, (1280, 720), interpolation=cv2.INTER_AREA)
        # cv2.imwrite("FinalScreen.jpg", FinalScreen)
        return FinalScreen
    def forward(self, img):
        out_img = np.copy(img)
        # img = self.calibration.undistort(img)
        img=out_img
        bird_eye = self.transform.forward(img)
        cv2.imwrite("bird_eye.jpg", bird_eye)
        thresholded = self.thresholding.forward(bird_eye)
        cv2.imwrite("thresholded.jpg", thresholded)
        thresholded_lane = self.lanelines.forward(thresholded)
        lane = self.transform.backward(thresholded_lane)

        black_with_lane= cv2.addWeighted(out_img, 1, lane, 0.6, 0)
        final = self.lanelines.plot(black_with_lane)
        #debug
        pipelined = self.pipeline(img, final, bird_eye, thresholded, thresholded_lane, lane)
        return pipelined

    def forward_without_debug(self, img):
        out_img = np.copy(img)
        # img = self.calibration.undistort(img)
        img=out_img
        bird_eye = self.transform.forward(img)
        cv2.imwrite("bird_eye.jpg", bird_eye)
        thresholded = self.thresholding.forward(bird_eye)
        cv2.imwrite("thresholded.jpg", thresholded)
        thresholded_lane = self.lanelines.forward(thresholded)
        lane = self.transform.backward(thresholded_lane)

        black_with_lane= cv2.addWeighted(out_img, 1, lane, 0.6, 0)
        final = self.lanelines.plot(black_with_lane)
        #debug
        return final

    def process_image(self,debug, input_path, output_path):
        img = mpimg.imread(input_path)
        if debug:
            out_img = self.forward(img)
        else:
            out_img = self.forward_without_debug(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self,debug, input_path, output_path):
        clip = VideoFileClip(input_path)
        if debug:
            out_clip = clip.fl_image(self.forward)
        else:
            out_clip = clip.fl_image(self.forward_without_debug)
        out_clip.write_videofile(output_path, audio=False)

args=sys.argv
if args[1] == "video":
    if args[2] == "0":
        f=FindLaneLines()
        f.process_video(0,args[3],args[4])
    elif args[2] == "1":
        f = FindLaneLines()
        f.process_video(1, args[3], args[4])
if args[1] == "image":
    if args[2] == "0":
        f=FindLaneLines()
        f.process_image(0,args[3],args[4])
    elif args[2] == "1":
        f = FindLaneLines()
        f.process_image(1, args[3], args[4])