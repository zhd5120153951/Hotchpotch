import cv2
import os
import threading
from threading import Lock, Thread

# 传入的视频路径和输出图像路径
# video_path = 'E:\\Source\\WorkSpace\\yolov5-7.0-fire\\728\\'
# img_path = 'E:\\Source\\WorkSpace\\yolov5-7.0-fire\\728'
# filelist = os.listdir(video_path)


def video2img(filename, i):
    # print(filename)
    cnt = 0
    dnt = 0
    # if os.path.exists(img_path + str(filename)):
    # pass
    # else:
    # os.mkdir(img_path + str(filename))
    cap = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        # width = frame.shape[1]
        # height = frame.shape[0]
        if (cnt % 25) == 0:  # 每隔25帧取一张图
            cv2.imwrite(img_path + '\\' + str(i)+'__'+str(dnt) +
                        '.jpg', frame)
            # cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[
            # 1].tofile(img_path)
            dnt += 1
            print(img_path + '\\' + str(i)+'__' + str(dnt) + '.jpg')
        cnt += 1
    cap.release()


if __name__ == '__main__':
    # 传入的视频路径和输出图像路径
    video_path = 'E:\\Datasets\\video_20240913\\主风机房值班室_20240914101213\\'
    img_path = 'E:\\Datasets\\video_20240913\\images'

    filelist = os.listdir(video_path)
    for i, filename in enumerate(filelist):
        file_path = ''.join([video_path, filename])
        print(file_path)
        threading.Thread(target=video2img, args=(file_path, i)).start()
        print('\nconvert video into img successfully')
