#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
from collections.abc import Callable, Iterable, Mapping
import os
import json
import time
from typing import Any
import cv2
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
from utils import COLORS, COCO_CLASSES
import logging
import threading
from queue import Queue
import multiprocessing as mp
import joblib

# logging.basicConfig(level=logging.INFO)#原始logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s-%(levelname)s-%(message)s')
# sail.set_print_flag(1)


class YOLOv5:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        print(self.net.get_graph_names()[0])
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        print(self.net.get_input_names(self.graph_name)[0])
        self.output_names = self.net.get_output_names(self.graph_name)
        print(self.net.get_output_names(self.graph_name))
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        if len(self.output_names) not in [1, 3]:
            raise ValueError('only suport 1 or 3 outputs, but got {} outputs bmodel'.format(len(self.output_names)))

        self.batch_size = self.input_shape[0]  #原本输入是一张图--分割成6份
        print("batch_size" + str(self.input_shape[0]))
        self.net_h = self.input_shape[2]
        print("net_h==input_h" + str(self.input_shape[2]))
        self.net_w = self.input_shape[3]
        print("net_w==input_w" + str(self.input_shape[3]))

        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.agnostic = False
        self.multi_label = True
        self.max_det = 1000

        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        letterbox_img, ratio, (tx1, ty1) = self.letterbox(ori_img,
                                                          new_shape=(self.net_h, self.net_w),
                                                          color=(114, 114, 114),
                                                          auto=False,
                                                          scaleFill=False,
                                                          scaleup=True,
                                                          stride=32)

        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.astype(np.float32)
        # input_data = np.expand_dims(input_data, 0)
        img = np.ascontiguousarray(img / 255.0)
        return img, ratio, (tx1, ty1)

    def letterbox(self,
                  im,
                  new_shape=(640, 640),
                  color=(114, 114, 114),
                  auto=False,
                  scaleFill=False,
                  scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def predict(self, input_img, img_num):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)

        # resort
        out_keys = list(outputs.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n == k:
                    ord.append(i)
                    break
        out = [outputs[out_keys[i]][:img_num] for i in ord]
        return out

    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])

        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)

        start_time = time.time()
        outputs = self.predict(input_img, img_num)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results


class RTSPThread(threading.Thread):
    def __init__(self, rtsp_url, queue) -> None:
        threading.Thread.__init__(self)
        self.rtsp_url = rtsp_url
        self.queue = queue

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.open(args.input):
            raise Exception("can't open the rtsp streaming")

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(fps, size)
        num = 0
        while True:
            if num % 25 != 0:
                ret, frame = cap.read()
                num += 1
                continue
            ret, frame = cap.read()
            if not ret:
                print("读取视频终止")
                cap = cv2.VideoCapture(args.input)
                if not cap.isOpened():
                    print("无法打开rtsp流")
                    continue
            self.queue.put(frame)
            if self.queue.qsize() > 10:
                print("大于10")
                self.queue.get()
            if self.queue.qsize() == 0:  #处理速度大于读取--也必须要让处理速度大于读取速度才不会出错
                print("等于0")
                continue
            if num % 25 == 0:
                num = 1
            # print(self.queue.qsize())
            # time.sleep(0.88)#延时1秒

        cap.release()


class YOLOv5Thread(threading.Thread):
    def __init__(self, queue, yolov5, output_img_dir, batch_size, svm):
        threading.Thread.__init__(self)
        self.queue = queue
        self.yolov5 = yolov5
        self.output_img_dir = output_img_dir
        self.batch_size = batch_size
        self.svm = svm

    def run(self):
        frame_list = []
        cn = 1
        # count = 1
        while True:
            if not self.queue.empty():
                frame = self.queue.get()
                # print("yolo"+str(self.queue.qsize()))
                # t1 = time.time()
                # cv2.imwrite(os.path.join(self.output_img_dir,str(count)+"_.jpg"),frame)
                # count+=1

                # for row in range(1,3):
                #     for col in range(1,4):
                #         frame_list.append(frame[440*(row-1):440*row+200,640*(col-1):640*col])

                for row in range(1, 4):
                    for col in range(1, 4):
                        frame_list.append(frame[360 * (row - 1):360 * row, 640 * (col - 1):640 * col])

                # cv2.imwrite(os.path.join(self.output_img_dir,str(count)+"_.jpg"),frame)
                # count+=1
                if len(frame_list) == self.batch_size:  #batch=6/9
                    results = self.yolov5(frame_list)
                    for i, _ in enumerate(frame_list):
                        det = results[i]

                        if det.shape[0] == 0:  #没检测到
                            # print(det.shape)
                            continue
                        for j in range(len(det[:, -1])):
                            # print(det[:4])#二维张量[[每张图检测到目标的xyxy]]
                            # print(det[j,:4].astype(np.uint16))
                            # print(det[j,:4][0].astype(np.uint16))
                            # print(det[:,-1])#一维张量[每张图检测到目标的类别--0fire,1--nofire]
                            # print(det[:,-2])#一维张量[每张图检测到目标的得分]
                            if (int(det[:, -1][j]) + 1) != 1:  #1--fire,2--nofire
                                # print(det[:,-1][j])
                                continue

                            if det[:, -2][j] >= 0.45:  #得分大于0.45的认为是火，否则加入二次判别
                                #只处理fire--模型判断还不行，加入颜色、运动等信息
                                logging.info("No.{},det nums:{}".format(cn, det.shape[0]))
                                res_frame = draw_numpy_fire(frame_list[i],
                                                            det[j, :4],
                                                            masks=None,
                                                            classes_ids=det[:, -1][j],
                                                            conf_scores=det[:, -2][j])
                                cv2.imwrite(os.path.join(self.output_img_dir, str(cn) + ".jpg"), res_frame)
                                cn += 1
                                #检测到火后判断运动
                            else:
                                #获取到该区域
                                img = frame_list[i][det[j, :4][1].astype(np.uint16):det[j, :4][3].astype(np.uint16),
                                                    det[j, :4][0].astype(np.uint16):det[j, :4][2].astype(np.uint16)]

                                size = list(img.shape)
                                size[2] = 1

                                pre = self.svm.predict(np.array(img).reshape(-1, 3))
                                pre = pre.reshape([size[0], size[1]])

                                kernel = np.ones((3, 3), dtype=np.uint8)
                                pre = cv2.dilate(cv2.erode(pre, kernel), kernel)
                                pre = pre * 255

                                pre = pre.astype(np.uint8)
                                contours, _ = cv2.findContours(pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for contour in contours:
                                    area = cv2.contourArea(contour)
                                    if area >= int(img.shape[0] * img.shape[1] * 0.75):  #设阈值控制
                                        print("大于阈值--二次确认fire")
                                        ret = cv2.drawContours(img, contour, -1, (0, 255, 0), 1)
                                        cv2.imwrite(os.path.join(self.output_img_dir, str(cn) + '_.jpg'), ret)
                                        res_frame = draw_numpy_fire(frame_list[i],
                                                                    det[j, :4],
                                                                    masks=None,
                                                                    classes_ids=det[:, -1][j],
                                                                    conf_scores=det[:, -2][j])
                                        cv2.imwrite(os.path.join(self.output_img_dir, str(cn) + '.jpg'), res_frame)
                                        cn += 1
                                    else:
                                        continue

                # t2 = time.time()-t1
                # print("每张图推理耗时：{}s".format(t2))
                frame_list.clear()


def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None):
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        logging.debug("class id={}, score={}, (x1={},y1={},x2={},y2={})".format(classes_ids[idx], conf_scores[idx], x1,
                                                                                y1, x2, y2))
        if conf_scores[idx] < 0.25:
            continue
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(image,
                        COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(conf_scores[idx], 2)), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255),
                        thickness=1)
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5

    return image


#重载绘图函数--传入一张图的待画参数
def draw_numpy_fire(image, boxes, masks=None, classes_ids=None, conf_scores=None):
    x1, y1, x2, y2 = boxes[:].astype(np.int32).tolist()
    logging.debug("class id={}, score={}, (x1={},y1={},x2={},y2={})".format(classes_ids, conf_scores, x1, y1, x2, y2))
    if conf_scores < 0.25:
        return

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
    if classes_ids is not None and conf_scores is not None:
        classes_ids = classes_ids.astype(np.int8)
        cv2.putText(image,
                    COCO_CLASSES[classes_ids + 1] + ':' + str(round(conf_scores, 2)), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0),
                    thickness=1)
        if masks is not None:
            mask = masks[:, :, :]
            image[mask] = image[mask] * 0.5 + np.array((0, 255, 0)) * 0.5

    return image


def main(args):
    # check params
    if not os.path.exists(args.input):
        if args.input.startswith('rtsp://'):
            print("using rtsp as video source")  #debug时不用raise()
        else:
            raise FileNotFoundError('{} is not existed.'.format(args.input))  #命令行执行时用raise()
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    if not os.path.exists(args.pkl):
        raise FileNotFoundError('{} is not existed.'.format(args.pkl))
        # print("pkl模型文件不存在")

    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)

    # initialize net
    yolov5 = YOLOv5(args)
    batch_size = yolov5.batch_size

    # warm up
    # for i in range(10):
    #     results = yolov5([np.zeros((640, 640, 3))])
    yolov5.init()

    # load svm
    svm = joblib.load(args.pkl)

    # test video--应该区分本地视频和rtsp视频
    if os.path.isfile(args.input):  #本地视频是逐帧检测
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can not open the video")
        #if not cap.open("rtsp://admin:1QAZ@wsx@192.168.21.251:554/Streaming/Channels/101"):
        #    raise Exception("can't read the rtsp streaming")
        #if not cap.open("rtsp://admin:jiankong123@192.168.23.10:554/Streaming/Channels/101"):
        #    raise Exception("can't open the rtsp streaming")

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(fps, size)
        # save_video = os.path.join(output_dir, os.path.splitext(os.path.split(args.input)[1])[0] + '.avi')
        # out = cv2.VideoWriter(save_video, fourcc, fps, size)  #保存原视频---长时间测试时不要
        cn = 1
        # numCount = 0
        frame_list = []
        while True:
            # if numCount % 25 != 0:
            #     numCount += 1
            #     continue
            ret, frame = cap.read()

            if not ret or frame is None:
                break
            for row in range(1, 4):  #9个batch
                for col in range(1, 4):
                    frame_list.append(frame[360 * (row - 1):360 * row, 640 * (col - 1):640 * col])

            if len(frame_list) == batch_size:
                results = yolov5(frame_list)
                for i, _ in enumerate(frame_list):
                    det = results[i]
                    if det.shape[0] == 0:  #没检测到
                        continue
                    for j in range(len(det[:, -1])):
                        if int(det[:, -1][j]) != 0:  #0--fire,1--nofire(以后单类别这里就不需要了)
                            continue
                        if det[:, -2][j] >= 0.5:  #得分大于0.5的认为是火，否则加入二次判别
                            #只处理fire--模型判断还不行，加入颜色、运动等信息
                            logging.info("No.{},det nums:{}".format(cn, det.shape[0]))
                            res_frame = draw_numpy_fire(frame_list[i],
                                                        det[j, :4],
                                                        masks=None,
                                                        classes_ids=det[:, -1][j],
                                                        conf_scores=det[:, -2][j])
                            cv2.imwrite(os.path.join(output_img_dir, str(cn) + ".jpg"), res_frame)
                            cn += 1
                        else:
                            #获取到该区域
                            img = frame_list[i][det[j, :4][1].astype(np.uint16):det[j, :4][3].astype(np.uint16),
                                                det[j, :4][0].astype(np.uint16):det[j, :4][2].astype(np.uint16)]

                            size = list(img.shape)
                            size[2] = 1

                            pre = svm.predict(np.array(img).reshape(-1, 3))  #二次预测
                            pre = pre.reshape([size[0], size[1]])

                            kernel = np.ones((3, 3), dtype=np.uint8)
                            pre = cv2.dilate(cv2.erode(pre, kernel), kernel)
                            pre = pre * 255

                            pre = pre.astype(np.uint8)
                            contours, _ = cv2.findContours(pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for contour in contours:
                                area = cv2.contourArea(contour)
                                if area >= (size[0] * size[1]) // 2:  #设阈值控制
                                    print("大于400--二次确认fire")
                                    ret = cv2.drawContours(img, contour, -1, (0, 255, 0), 1)
                                    cv2.imwrite(os.path.join(output_img_dir, str(cn) + '_.jpg'), ret)
                                    res_frame = draw_numpy_fire(frame_list[i],
                                                                det[j, :4],
                                                                masks=None,
                                                                classes_ids=det[:, -1][j],
                                                                conf_scores=det[:, -2][j])
                                    cv2.imwrite(os.path.join(output_img_dir, str(cn) + '.jpg'), res_frame)
                                    cn += 1
                                else:
                                    continue

                frame_list.clear()
        logging.info("本地推理结束...")
    else:  #网络rtsp--多线程

        rtsp_thread = RTSPThread(args.input, queue)
        yolov5_thread = YOLOv5Thread(queue, yolov5, output_img_dir, batch_size, svm)

        rtsp_thread.start()
        yolov5_thread.start()

        logging.info("双线程开始运行...")
    # average_latency = decode_time + preprocess_time + inference_time + postprocess_time
    # qps = 1 / average_latency
    # logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input',
                        type=str,
                        default='rtsp://admin:jiankong123@192.168.23.10:554/Streaming/Channels/101',
                        help='path of input')
    parser.add_argument('--bmodel',
                        type=str,
                        default='./models/BM1684/yolov5-fire-821-onnx-batch9.bmodel',
                        help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--pkl', type=str, default='./models/BM1684/SVC.pkl', help='SVM model')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    # args.input = './datasets/test/1.mp4'
    queue = Queue(maxsize=10)  #最大队列数--线程
    main(args)
    # logging.info("result saved in images")
