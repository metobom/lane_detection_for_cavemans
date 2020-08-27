import cv2 
import numpy as np


class lane_detector():
    def __init__(self):
        self.vid = cv2.VideoCapture('input_vid.mp4')
        self.canny_up_th = 150
        self.canny_low_th = self.canny_up_th / 3; # as recommended

    def prepare_frame(self, frame):
        frame = np.mean(frame, axis = 2).astype(np.uint8) # cvt gray
        frame = cv2.GaussianBlur(frame, (5,5), 0) # smooth frame
        frame = cv2.Canny(frame, self.canny_low_th, self.canny_up_th)
        return frame
    
    #region of interest
    def roi(self, frame):
        h = frame.shape[0] 
        #points are -> corners of polygon (picked them manually)
        roi = np.array([[(100, h), (1258, h), (783, 420), (547, 408)]]) # interested in a polygon area, which is required fov of camera.
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, roi, 255) # create black img in shape of frame. fill roi with 255 pix val.
        separated_roi = cv2.bitwise_and(frame, mask) # only roi will appear in image. Other regions are black
        return separated_roi
    
    def detect_lines(self, frame):
        #param1: separated_roi, param23: 2 (rho) to 1 (teta), param4: threshold. when a bin val is 100 (100 intersections in hough space) there is a line.
        #param5: placeholder, param67: obvious
        #play with params to get better results for your input vid
        return cv2.HoughLinesP(frame, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5) 
    
    def drawLines(self, frame, lines):
        frame_bl = np.zeros_like(frame)
        if lines is not None: #just in case
            for line in lines:
                u0, v0, u1, v1 = line.reshape(4) # get start and end points of hough lines
                print(u0, v0, u1, v1)
                cv2.line(frame_bl, (u0, v0), (u1, v1), (0, 125, 255), 5)
        return frame_bl

    def main(self):
        while True:
            check, frame = self.vid.read()
            if check:
                frame = cv2.resize(frame, (1280, 720)) # I have a small screen :(
                original_frame = np.copy(frame)
                frame = self.prepare_frame(frame)
                frame = self.roi(frame)
                lines = self.detect_lines(frame)
                black_bg_line_img = self.drawLines(original_frame, lines) # every pix is black instead of drawn lines
                #blend color and black img. param1: color img. param2: multip weight of color img. param3: black img.
                #param4: multip weight of black img. param5: gamma
                frame = cv2.addWeighted(original_frame, 0.7, black_bg_line_img, 0.7, 1)
                cv2.imshow('frame', frame)
                cv2.waitKey(1)

if __name__ == '__main__':
    detector = lane_detector()
    detector.main()

