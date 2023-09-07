import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("test.mp4")
mog = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if ret == True:
        fmask = mog.apply(frame)
        cv2.imshow("test", fmask)
    key = cv2.waitKey(1)
    #等待用户按下案件
    if (key == 27):
        break
cap.release()
cv2.destroyAllWindows()
#接下来时找轮廓步骤