import cv2
import random
import os
import glob


def renameImg(path_orig,path_dst):
    imgList = os.listdir(path_orig)
    for img in imgList:
        if img.endswith(".jpg"):
            name = img.split(".",3)[0]+"."+img.split(".",3)[1]
            src = os.path.join(os.path.abspath(path_orig),img)
            dst = os.path.join(os.path.abspath(path_dst),name+".jpg")
            try:
                os.rename(src,dst)
            except:
                continue
    

if __name__ == "__main__":
    renameImg()