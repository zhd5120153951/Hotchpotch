import cv2
import os
import threading
from threading import Lock, Thread

#传入的视频路径和输出图像路径
video_path = 'E:\\Source\\WorkSpace\\yolov5-7.0-fire\\728\\'
img_path = 'E:\\Source\\WorkSpace\\yolov5-7.0-fire\\728'
filelist = os.listdir(video_path)


def video2img(filename):
    print(filename)
    cnt = 0
    dnt = 0
    # if os.path.exists(img_path + str(filename)):
    # pass
    # else:
    # os.mkdir(img_path + str(filename))
    cap = cv2.VideoCapture(video_path + str(filename))
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

<<<<<<< HEAD
        # width = frame.shape[1]
        # height = frame.shape[0]
        if (cnt % 25) == 0:  # 每隔25帧取一张图
            cv2.imwrite(img_path + '\\' + str(i)+'__'+str(dnt) +
                        '.jpg', frame)
=======
        width = frame.shape[1]
        height = frame.shape[0]
        if (cnt % 25) == 0:  #每隔5帧取一张图
            cv2.imwrite(img_path + '\\' + str(dnt) + '.jpg', frame)
>>>>>>> 0238ffee67d85c503aa2b9699c52ac3e6b2206ac
            dnt += 1
            # cv2.imencode('.jpg', frame[1].tofile(img_path + str(filename) + '\\' + str(dnt) + '.jpg'))
            print(img_path + '\\' + str(dnt) + '.jpg')
        cnt += 1
        if cv2.waitKey(1) % 0xFF == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
<<<<<<< HEAD
    # 传入的视频路径和输出图像路径
    video_path = 'F:\\WorkPlace\\yolov5-tensorrt\\smoke_video\\'
    img_path = 'F:\\DataSet\\zhongkewubao\\smoke_wubao_v2'

    filelist = os.listdir(video_path)
    for i, filename in enumerate(filelist):
=======
    for filename in filelist:
>>>>>>> 0238ffee67d85c503aa2b9699c52ac3e6b2206ac
        print(filename)
        threading.Thread(target=video2img, args=(filename, )).start()
        print('\nconvert video into img successfully')