'''
@FileName   :video2img.py
@Description:
@Date       :2023/08/31 16:19:31
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
import os
import threading
from threading import Lock, Thread

#传入的视频路径和输出图像路径
video_path = 'D:\\FilePackage\\BaiduDiskDownload\\Video\\'
img_path = 'D:\\FilePackage\\BaiduDiskDownload\\Video\\27'
filelist = os.listdir(video_path)


def video2img(filename):
    cnt = 0
    dnt = 1
    # if os.path.exists(img_path + str(filename)):
    # pass
    # else:
    # os.mkdir(img_path + str(filename))
    cap = cv2.VideoCapture(video_path + str(filename))
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        # width = frame.shape[1]
        # height = frame.shape[0]
        if (cnt % 12) == 0:  #每隔25帧取一张图
            cv2.imwrite(img_path + '\\' + str(dnt) + '_27.jpg', frame)
            dnt += 1
            # cv2.imencode('.jpg', frame[1].tofile(img_path + str(filename) + '\\' + str(dnt) + '.jpg'))
            print(img_path + '\\' + str(dnt) + '.jpg')
        cnt += 1
        if cv2.waitKey(1) % 0xFF == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    for filename in filelist:
        print(filename)
        threading.Thread(target=video2img, args=(filename, )).start()
        print('\nconvert video into img successfully')