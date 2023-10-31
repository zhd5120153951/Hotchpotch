#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json
import time
import cv2
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
from utils import COLORS, COCO_CLASSES
import logging
import multiprocessing as mp
import joblib
import requests
import base64
import datetime

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s-%(levelname)s-%(message)s')
# sail.set_print_flag(1)


class YOLOv5:

    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        if len(self.output_names) not in [1, 3]:
            raise ValueError('only suport 1 or 3 outputs, but got {} outputs bmodel'.format(len(self.output_names)))

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

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
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(image,
                        COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(conf_scores[idx], 2)), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        thickness=2)
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5

    return image


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


#图像编码
def img2base64(img_path):
    # 读取图片文件
    with open(img_path, 'rb') as image_file:
        # 将图片内容进行base64编码
        encoded_image = base64.b64encode(image_file.read())

        return encoded_image.decode("utf-8")  #把byte转换为字符串


#推理函数
def RegionDetect(args, queue, output_img_dir):
    # initialize net
    yolov5 = YOLOv5(args)
    batch_size = yolov5.batch_size
    yolov5.init()

    #获取区域点
    json_data = read_json('../region/region.json')
    pts = []
    for pt in json_data:
        pts.append(json_data[pt])
    pts = np.array(pts)

    cn = 0
    frame_list = []
    while True:
        frame = queue.get()
        # start_time=time.time()
        #区域入侵--盒子上无法直接显示图像没法直接标定区域--暂用固定点代替
        temp_frame = frame.copy()
        mask = np.zeros([temp_frame.shape[0], temp_frame.shape[1]], dtype=np.uint8)
        # 固定区域
        # pts = np.array([
        #     [int(temp_frame.shape[1]*0.2),int(temp_frame.shape[0]*0.1)],
        #     [int(temp_frame.shape[1]*0.7),int(temp_frame.shape[0]*0.12)],
        #     [int(temp_frame.shape[1]*0.56),int(temp_frame.shape[0]*0.75)],
        #     [int(temp_frame.shape[1]*0.14),int(temp_frame.shape[0]*0.65)]
        # ],np.int32)
        mask = cv2.fillPoly(mask, [pts], (255, 255, 255))
        # imgc  =frame.transpose((0,1,2))#改变通道分布
        # imgc = cv2.add(imgc,np.zeros(np.shape(imgc),dtype=np.uint8),mask=mask)
        temp_frame = cv2.add(temp_frame, np.zeros(np.shape(temp_frame), dtype=np.uint8), mask=mask)
        # frame = imgc.transpose((2,0,1))
        # frame = imgc.transpose((0,1,2))

        frame_list.append(temp_frame)
        if len(frame_list) == batch_size:
            results = yolov5(frame_list)
            for i, _ in enumerate(frame_list):
                det = results[i]
                if det.shape[0] == 0:
                    continue
                cn += 1
                logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                res_frame = draw_numpy(frame, det[:, :4], masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2])
                cv2.polylines(res_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
                cv2.imwrite(os.path.join(output_img_dir, str(cn) + '.jpg'), res_frame)
                curTime = datetime.datetime.now()
                curTime = curTime.strftime("%Y-%m-%d %H:%M:%S")
                #编码
                encoded_string = img2base64(os.path.join(output_img_dir, str(cn) + '.jpg'))
                data = {'device': 'hikvision', 'date': curTime, 'img': encoded_string}
                header = {"Content-Type": "application/json; charset=utf-8"}
                try:
                    ret = requests.post("http://192.168.22.4:6666", json.dumps(data), headers=header)
                    logging.info(ret.text)
                except Exception as e:
                    logging.error(e)

                if cn == 200:  #最多保存200张图本地--之后覆盖前面
                    cn = 0
            frame_list.clear()
        # end_time = time.time()-start_time
        # print('单次推理用时：',end_time)


#获取图像帧
def frame_put(q, rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        logging.info('{} open successed.'.format(rtsp_url))
    else:
        logging.error('{} open failed.'.format(rtsp_url))
    num = 0
    while True:
        if num % 8 != 0:
            frame = cap.read()
            num += 1
            continue
        q.put(cap.read()[1])

        if q.qsize() > 10:
            # print('q>10')
            logging.warning('queue size over 10...')
            q.get()
        if q.qsize() == 0:
            # print('q==0')
            logging.warning('queue size equal 10...')
            continue
        if num % 8 == 0:
            num = 1


def main(args):
    # check params
    if not os.path.exists(args.input):
        if args.input.startswith('rtsp://'):
            # print('using rtsp as video source')
            logging.debug('using rtsp as video source')
        else:
            logging.warning('{} is not existed.'.format(args.input))
            raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        logging.warning('{} is not existed.'.format(args.bmodel))
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))

    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)

    # initialize net
    # yolov5 = YOLOv5(args)
    # batch_size = yolov5.batch_size

    # warm up
    # for i in range(10):
    #     results = yolov5([np.zeros((640, 640, 3))])
    # yolov5.init()

    # decode_time = 0.0
    # test local video
    if os.path.isfile(args.input):
        pass  #屏蔽本地推理
    # test rtsp video
    else:
        mp.set_start_method(method='spawn')  #init
        queue = mp.Queue(maxsize=10)  #一个摄像头

        processes = []

        processes.append(mp.Process(target=RegionDetect, args=(args, queue, output_img_dir)))

        processes.append(mp.Process(target=frame_put, args=(queue, args.input)))

        for process in processes:
            process.daemon = True
            process.start()
        for process in processes:
            process.join()

        # cap = cv2.VideoCapture()
        # # if not cap.open(args.input):
        # #    raise Exception("can not open the video")
        # if not cap.open(args.input):
        #     logging.error("can't read the rtsp streaming")
        #     raise Exception("can't read the rtsp streaming")

        # # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print(fps, size)
        # # save_video = os.path.join(output_dir, os.path.splitext(os.path.split(args.input)[1])[0] + '.avi')
        # # out = cv2.VideoWriter(save_video, fourcc, fps, size)
        # cn = 0
        # frame_list = []
        # json_data = read_json('./region/region.json')
        # pts = []
        # for pt in json_data:
        #     pts.append(json_data[pt])
        # pts = np.array(pts)
        # while True:
        #     ret, frame = cap.read()
        #     if not ret or frame is None:
        #         break

        #     #区域入侵--盒子上无法直接显示图像没法直接标定区域--暂用固定点代替
        #     temp_frame = frame.copy()
        #     mask = np.zeros([temp_frame.shape[0],temp_frame.shape[1]],dtype=np.uint8)
        #     # 固定区域
        #     # pts = np.array([
        #     #     [int(temp_frame.shape[1]*0.2),int(temp_frame.shape[0]*0.1)],
        #     #     [int(temp_frame.shape[1]*0.7),int(temp_frame.shape[0]*0.12)],
        #     #     [int(temp_frame.shape[1]*0.56),int(temp_frame.shape[0]*0.75)],
        #     #     [int(temp_frame.shape[1]*0.14),int(temp_frame.shape[0]*0.65)]
        #     # ],np.int32)
        #     mask = cv2.fillPoly(mask,[pts],(255,255,255))
        #     # imgc  =frame.transpose((0,1,2))#改变通道分布
        #     # imgc = cv2.add(imgc,np.zeros(np.shape(imgc),dtype=np.uint8),mask=mask)
        #     temp_frame = cv2.add(temp_frame,np.zeros(np.shape(temp_frame),dtype=np.uint8),mask=mask)
        #     # frame = imgc.transpose((2,0,1))
        #     # frame = imgc.transpose((0,1,2))

        #     frame_list.append(temp_frame)
        #     if len(frame_list) == batch_size:
        #         results = yolov5(frame_list)
        #         for i, _ in enumerate(frame_list):
        #             det = results[i]
        #             if det.shape[0] == 0:
        #                 continue
        #             cn += 1
        #             logging.info("{}, det nums: {}".format(cn, det.shape[0]))
        #             res_frame = draw_numpy(frame, det[:,:4], masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2])
        #             cv2.polylines(res_frame,[pts],isClosed=True,color=(0,255,0),thickness=1)
        #             cv2.imwrite(os.path.join(output_img_dir,str(cn)+'.jpg'),res_frame)
        #             # out.write(res_frame)
        #         frame_list.clear()
        #     # end_time = time.time()-start_time
        #     # print('单次推理用时：',end_time)


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input',
                        type=str,
                        default='rtsp://admin:jiankong123@192.168.23.15:554/Streaming/Channels/101',
                        help='path of input')
    parser.add_argument('--bmodel',
                        type=str,
                        default='../models/BM1684/yolov5s-regiondetect-onnx-batch1.bmodel',
                        help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)