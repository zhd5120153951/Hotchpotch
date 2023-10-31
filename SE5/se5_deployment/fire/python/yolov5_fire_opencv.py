#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import json
import time
import cv2
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
from utils import COLORS, COCO_CLASSES
import logging
import threading
from queue import Queue

logging.basicConfig(level=logging.INFO)
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
        print("batch_size-" + str(self.input_shape[0]))
        self.net_h = self.input_shape[2]
        print("net_h==" + str(self.input_shape[2]))
        self.net_w = self.input_shape[3]
        print("net_w==" + str(self.input_shape[3]))

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

def draw_numpy_fire(image, boxes, masks=None, classes_ids=None, conf_scores=None):
    x1,y1,x2,y2 = boxes[:].astype(np.int32).tolist()
    logging.debug("class id={}, score={}, (x1={},y1={},x2={},y2={})".format(classes_ids, conf_scores, x1, y1, x2, y2))
    if conf_scores < 0.15:
        return
    
    cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),thickness=1)
    if classes_ids is not None and conf_scores is not None:
        classes_ids = classes_ids.astype(np.int8)
        cv2.putText(image,
                    COCO_CLASSES[classes_ids + 1] + ':'
                    + str(round(conf_scores,2)),(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),thickness=1)
        if masks is not None:
            mask = masks[:, :,:]
            image[mask] = image[mask] * 0.5 + np.array((0,0,255)) * 0.5

    return image


def main(args):
    # check params
    if not os.path.exists(args.input):
        print(args.input)
        # raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))

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

    decode_time = 0.0
    # test images--参数是文件目录
    if os.path.isdir(args.input):
        img_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg', '.png', '.jpeg', '.bmp', '.webp']:
                    continue
                img_file = os.path.join(root, filename)
                cn += 1
                logging.info("{}, img_file: {}".format(cn, img_file))
                # decode
                start_time = time.time()
                src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
                if src_img is None:
                    logging.error("{} imdecode is None.".format(img_file))
                    continue
                if len(src_img.shape) != 3:
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                decode_time += time.time() - start_time

                img_list.append(src_img)
                filename_list.append(filename)
                if len(img_list) == batch_size:
                    # predict
                    results = yolov5(img_list)

                    for i, filename in enumerate(filename_list):
                        det = results[i]
                        if det.shape[0] == 0:
                            continue
                        for j in range(len(det[:,-1])):
                                # print(det[j,:4])#二维张量[[每张图检测到目标的xyxy]]
                                # print(det[:,-1])#一维张量[每张图检测到目标的类别--0fire,1--nofire]
                                # print(det[:,-2])#一维张量[每张图检测到目标的得分]
                                if int(det[:,-1][j]) != 0:#0--fire,1--nofire
                                    #print(COCO_CLASSES[int(det[:,-1][j])+1])
                                    continue      
                                # save image
                                res_img = draw_numpy_fire(img_list[i],
                                                    det[j, :4],
                                                    masks=None,
                                                    classes_ids=det[:, -1][j],
                                                    conf_scores=det[:, -2][j])
                                cv2.imwrite(os.path.join(output_img_dir, filename), res_img)

                        # save result
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['bboxes'] = []
                        for idx in range(det.shape[0]):
                            bbox_dict = dict()
                            x1, y1, x2, y2, score, category_id = det[idx]
                            bbox_dict['bbox'] = [
                                float(round(x1, 3)),
                                float(round(y1, 3)),
                                float(round(x2 - x1, 3)),
                                float(round(y2 - y1, 3))
                            ]
                            bbox_dict['category_id'] = int(category_id)
                            bbox_dict['score'] = float(round(score, 5))
                            res_dict['bboxes'].append(bbox_dict)
                        results_list.append(res_dict)

                    img_list.clear()
                    filename_list.clear()

        if len(img_list):
            results = yolov5(img_list)
            for i, filename in enumerate(filename_list):
                det = results[i]
                res_img = draw_numpy(img_list[i],
                                     det[:, :4],
                                     masks=None,
                                     classes_ids=det[:, -1],
                                     conf_scores=det[:, -2])
                cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
                res_dict = dict()
                res_dict['image_name'] = filename
                res_dict['bboxes'] = []
                for idx in range(det.shape[0]):
                    bbox_dict = dict()
                    x1, y1, x2, y2, score, category_id = det[idx]
                    bbox_dict['bbox'] = [
                        float(round(x1, 3)),
                        float(round(y1, 3)),
                        float(round(x2 - x1, 3)),
                        float(round(y2 - y1, 3))
                    ]
                    bbox_dict['category_id'] = int(category_id)
                    bbox_dict['score'] = float(round(score, 5))
                    res_dict['bboxes'].append(bbox_dict)
                results_list.append(res_dict)
            img_list.clear()
            filename_list.clear()

        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(
            args.input)[-1] + "_opencv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            # json.dump(results_list, jf)
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    # test video--应该区分本地视频和rtsp视频
    elif os.path.isfile(args.input):  #本地视频是逐帧检测
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can not open the video")
        #if not cap.open("rtsp://admin:1QAZ@wsx@192.168.21.251:554/Streaming/Channels/101"):
        #    raise Exception("can't read the rtsp streaming")
        #if not cap.open("rtsp://admin:jiankong123@192.168.23.10:554/Streaming/Channels/101"):
        #    raise Exception("can't open the rtsp streaming")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(fps, size)
        # save_video = os.path.join(output_dir, os.path.splitext(os.path.split(args.input)[1])[0] + '.avi')
        # out = cv2.VideoWriter(save_video, fourcc, fps, size)  #保存原视频---长时间测试时不要
        cn = 0
        frame_list = []
        while True:
            # start_time = time.time()
            
            ret, frame = cap.read()
            # decode_time += time.time() - start_time  #解码时长
            if not ret or frame is None:
                break
            for row in range(1,3):
                for col in range(1,4):
                    frame_list.append(frame[440*(row-1):440*row+200,640*(col-1):640*col])
            # frame_list.append(frame)
            if len(frame_list) == batch_size:
                results = yolov5(frame_list)
                for i, _ in enumerate(frame_list):
                    det = results[i]
                    if det.shape[0] == 0:
                        continue
                    for j in range(len(det[:,-1])):
                        if int(det[:,-1][j]) != 0:
                            print(det[:,-1][j])
                            continue
                        logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                        res_frame = draw_numpy_fire(frame_list[i],
                                            det[j, :4],
                                            masks=None,
                                            classes_ids=det[:, -1][j],
                                            conf_scores=det[:, -2][j])
                        cv2.imwrite(os.path.join(output_img_dir, str(cn)+'.jpg'), res_frame)
                        cn+=1
                        # out.write(res_frame)
            frame_list.clear()
        if len(frame_list):
            results = yolov5(frame_list)
            for i, frame in enumerate(frame_list):
                det = results[i]
                cn += 1
                logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                res_frame = draw_numpy(frame_list[i],
                                       det[:, :4],
                                       masks=None,
                                       classes_ids=det[:, -1],
                                       conf_scores=det[:, -2])
                # out.write(res_frame)
        cap.release()
        # out.release()
        # logging.info("result saved in {}".format(save_video))

        # calculate speed
        logging.info("------------------ Predict Time Info ----------------------")
        decode_time = decode_time / cn
        preprocess_time = yolov5.preprocess_time / cn
        inference_time = yolov5.inference_time / cn
        postprocess_time = yolov5.postprocess_time / cn
        logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
        logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
        logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
        logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    else:  #网络rtsp
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can't open the rtsp streaming")

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(fps, size)
        num = 1
        cn = 1
        frame_list = []
        
        while True:
            try:
                if num % 20 != 0:
                    num += 1
                    ret,frame = cap.read()
                    continue
               
                ret, frame = cap.read()
    
                # if not ret:
                #     cap = cv2.VideoCapture()
                #     if not cap.open(args.input):
                #         print("视频流解码失败")
                #         continue
                #     ret,frame = cap.read()
                if ret:
                    for row in range(1,3):
                        for col in range(1,4):
                            frame_list.append(frame[440*(row-1):440*row+200,640*(col-1):640*col])
                    
                    #cv2.imwrite(os.path.join(output_img_dir,str(count)+'_.jpg'),frame)
                    #count+=1
                    # frame_list.append(frame)
                    # print(frame_list[0].shape)
                    if len(frame_list) == 6:  #batch_size:
                        results = yolov5(frame_list)
                        # logging.info("No.{},detect nums:{}".format(num + 1, det.shape[0]))
                        # for i, _ in enumerate(frame_list):
                        #     det = results[i]
                            
                        #     if det.shape[0] == 0:  #没检测到
                        #         continue
                        #     for j in range(len(det[:,-1])):
                        #         # print(det[j,:4])#二维张量[[每张图检测到目标的xyxy]]
                        #         # print(det[:,-1])#一维张量[每张图检测到目标的类别--0fire,1--nofire]
                        #         # print(det[:,-2])#一维张量[每张图检测到目标的得分]
                        #         if int(det[:,-1][j]) != 0:#0--fire,1--nofire
                        #             #print(COCO_CLASSES[int(det[:,-1][j])+1])
                        #             continue                            
                        #         # logging.info("No.{}, det nums: {}".format(cn, det.shape[0]))
                        #         # logging.info("no.{},det nums:{}".format(cn,len(det[:,-1])))
                        #         res_frame = draw_numpy_fire(frame_list[i],
                        #                             det[j, :4],
                        #                             masks=None,
                        #                             classes_ids=det[:, -1][j],
                        #                             conf_scores=det[:, -2][j])
                        #         cv2.imwrite(os.path.join(output_img_dir, str(cn) + ".jpg"), res_frame)
                        #         cn+=1
                    # dt = time.time()-start_time
                    # print(dt)
                    frame_list.clear()
                    # if num % 25 == 0:
                    #     num = 1
                else:
                    # cap.release()
                    cap.release()
                    cap = cv2.VideoCapture(args.input)
                    if not cap.isOpened():
                        print("重连失败")
                    else:
                        print("重连成功")
                     
            except Exception as e:
                print(e)


        # cap.release()
        # logging.info("result saved in images")

    # average_latency = decode_time + preprocess_time + inference_time + postprocess_time
    # qps = 1 / average_latency
    # logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input',
                        type=str,
                        default='rtsp://admin:jiankong123@192.168.23.10:554/Streaming/Channels/101',
                        help='path of input')
    parser.add_argument('--bmodel', type=str, default='./models/BM1684/yolov5s-fire-727-batch6.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    # args.input = './datasets/test'
    # args.input = './datasets/816.mp4'
    main(args)
    print('all done.')
