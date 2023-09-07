from multiprocessing.connection import wait
import cv2
import matplotlib as plt
import numpy as np

img = cv2.imread("./1.jpg")
while True:
    cv2.imshow("person", img)
    if cv2.waitKey(0) & 0xff == ord("s"):
        cv2.imwrite("./123.png", img)
    elif cv2.waitKey(0) & 0xff == ord("q"):
        break
    else:
        cv2.destroyAllWindows()
