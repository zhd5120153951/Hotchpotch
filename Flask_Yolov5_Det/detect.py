'''
@FileName   :detect.py
@Description:
@Date       :2022/10/18 13:24:02
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import torch
import numpy as np
from models.experimental import attempt_load
from utils.dataloaders import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import cv2
from random import randint


class VideoCamera(object):
    def __init__(self):
        #通过opencv获取实时视频流
        self.img_size = 640
        self.threshold = 0.4
        self.max_frame = 160

        self.video = cv2.VideoCapture("data/1.mp4")  #换成自己的视频文件

        self.weights = 'yolov5s.pt'  #yolov5权重文件

        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)

        model.to(self.device).eval()
        # model.half()

        model.float()
        # torch.save(model, 'test.pt')
        self.m = model
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names]

    def __del__(self):
        self.video.release()

    def get_frame(self):
        while True:
            ret, frame = self.video.read()  #读
            if not ret:
                break
            im0, img = self.preprocess(frame)  #预处理

            pred = self.m(img, augment=False)[0]  #输入到模型
            pred = pred.float()
            pred = non_max_suppression(pred, self.threshold, 0.3)

            pred_boxes = []
            image_info = {}
            count = 0
            for det in pred:
                if det is not None and len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                    for *x, conf, cls_id in det:
                        label = self.names[int(cls_id)]
                        x1, y1 = int(x[0]), int(x[1])
                        x2, y2 = int(x[2]), int(x[3])
                        pred_boxes.append((x1, y1, x2, y2, label, conf))
                        count += 1
                        key = '{}-{:.02}'.format(label, count)
                        image_info[key] = ['{}x{}'.format(x2 - x1, y2 - y1), np.round(float(conf), 3)]

            frame = self.plot_bboxes(frame, pred_boxes)

            # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  #HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        # img = img.half()#半精度推理
        img = img.float()  #半精度化
        img /= 255.0  #图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        tl = line_thickness or round(0.002 * (image.sahpe[0] + image.shape[1]) / 2) + 1  #line/font thickness
        for (x1, y1, x2, y2, cls_id, conf) in bboxes:
            color = self.colors[self.names.index(cls_id)]
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

            tf = max(tl - 1, 1)  #font thickness
            t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  #filled
            cv2.putText(image,
                        '{}-{:.2f}'.format(cls_id, conf), (c1[0], c1[1] - 2),
                        0,
                        tl / 3, [255, 255, 255],
                        thickness=tf,
                        lineType=cv2.LINE_AA)
        return image
