import numpy as np
import cv2
import time


class knnDetector:
    def __init__(self, history, dist2Threshold, minArea):
        self.minArea = minArea
        # 混合高斯背景建模
        # self.detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
        self.detector = cv2.createBackgroundSubtractorKNN(history, dist2Threshold, False)
        self.kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detectOneFrame(self, frame):
        if frame is None:
            return None
        # start = time.time()
        mask = self.detector.apply(frame)
        # stop = time.time()
        # print("detect cast {} ms".format(stop - start))

        # start = time.time()
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel3)
        # stop = time.time()
        # print("open contours cast {} ms".format(stop - start))

        # start = time.time()
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # stop = time.time()
        # print("find contours cast {} ms".format(stop - start))
        i = 0
        bboxs = []
        # start = time.time()
        for c in contours:
            i += 1
            if cv2.contourArea(c) < self.minArea:
                continue

            bboxs.append(cv2.boundingRect(c))
        # stop = time.time()
        # print("select cast {} ms".format(stop - start))

        return mask, bboxs
