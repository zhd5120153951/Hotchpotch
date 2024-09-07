"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

#!/usr/bin/python
# -*- coding:utf8 -*-

from __future__ import print_function

import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from kalmanFilter import KalmanFilter

import cv2
import math

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IUO between two bboxes in the form [l,t,w,h]
    每一行代表一个跟踪框，每一列代表一个检测框，那么每个坐标的意义就是 跟踪框y与检测框x的IOU
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    # 这里用到了 maximum 的广播属性，从 44*1 × 1*56 广播到 44*56
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
     此类表示作为bbox观察到的单个跟踪对象的内部状态。
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        使用初始边界框初始化跟踪器。
        """
        # define constant velocity model
        # 这里dim_x=9 分别为 x, y, s, r, vx, vy, vs, ax, ay (vs指的是面积变化的速率)
        # dim_z=4 分别为 x, y, s, r
        self.kf = KalmanFilter(dim_x=9, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0, 0.5, 0],
             [0, 1, 0, 0, 0, 1, 0, 0, 0.5],
             [0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.org_box = bbox.copy()
        self.is_throw = False

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        update(bbox)：使用观测到的目标框bbox更新状态更新向量x(状态变量x)
        1.time_since_update = 0
                1.连续预测的次数，每执行predict一次即进行time_since_update+=1。
                2.在连续预测(连续执行predict)的过程中，一旦执行update的话，time_since_update就会被重置为0。
                2.在连续预测(连续执行predict)的过程中，只要连续预测的次数time_since_update大于0的话，
                    就会把hit_streak(连续更新的次数)重置为0，表示连续预测的过程中没有出现过一次更新状态更新向量x(状态变量x)的操作，
                    即连续预测的过程中没有执行过一次update。
        2.history = []      
                清空history列表。
                history列表保存的是单个目标框连续预测的多个结果([x,y,s,r]转换后的[x1,y1,x2,y2])，一旦执行update就会清空history列表。
        3.hits += 1：
                该目标框进行更新的总次数。每执行update一次，便hits+=1。
        4.hit_streak += 1
                1.连续更新的次数，每执行update一次即进行hit_streak+=1。
                2.在连续更新(连续执行update)的过程中，一旦开始连续执行predict两次或以上的情况下，
                    当连续第一次执行predict时，因为time_since_update仍然为0，并不会把hit_streak重置为0，
                    然后才会进行time_since_update+=1；
                    当连续第二次执行predict时，因为time_since_update已经为1，那么便会把hit_streak重置为0，
                    然后继续进行time_since_update+=1。
        5.kf.update(convert_bbox_to_z(bbox))
                convert_bbox_to_z负责将[x1,y1,x2,y2]形式的检测框转为滤波器的状态表示形式[x,y,s,r]，那么传入的为kf.update([x,y,s,r])。
                然后根据观测结果修改内部状态x(状态更新向量x)。
                使用的是通过yoloV3得到的“并且和预测框相匹配的”检测框来更新卡尔曼滤波器得到的预测框。

        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
         推进状态向量并返回预测的边界框估计值。
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        返回当前边界框估计值。
        Returns the current bounding box estimate.
        """
        bbox = convert_x_to_bbox(self.kf.x)[0] # 坐标转换，获得box的(xmin,ymin,xmax,ymax)
        # 计算估计框与修正框在x轴和y轴上的中心点偏差
        x = (bbox[0]+bbox[2])/2 - (self.org_box[0]+self.org_box[2])/2 
        y = (bbox[1] + bbox[3]) / 2 - (self.org_box[1] + self.org_box[3]) / 2
        disdance = math.hypot(x, y) # 获得两者的欧式距离
        # 业务处理：
        # 如果欧式距离大于两倍估计框的宽和修正框的宽和，同时大于估计框的高和修正框的高和，则认为是抛物行为
        # 主要基于第一：抛物行为类属于自由落体运动，速度会越来越快，
        # 第二：是摄像头的位置因为在下方，离摄像头位置越近，占据画面越大，从抛物行为的视角看，越往下，像素移动越多。
        # 默认为在高度方向上的变化大于横向上的。这个业务逻辑为了树叶、飞鸟等干扰项用，
        if disdance > 2 * (self.org_box[2]-self.org_box[0]+bbox[2]-bbox[0]) and \
            disdance > (self.org_box[3]-self.org_box[1]+bbox[3]-bbox[1]):
            self.is_throw = True

        return bbox, self.is_throw


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    将检测指定给跟踪对象（均表示为边界框）
    返回匹配项、unmatched_detections和unmatched_tracker的3个列表
    """
    #如果没有追踪到
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    #-----step1 获得 追踪器 和 检测目标 的 iou-----------------------------------
    iou_matrix = iou_batch(detections, trackers)
    
    #---------step2 获得匹配索引------------------------------------
    if min(iou_matrix.shape) > 0:
        # 将大于阈值的置为1，小于阈值的置为0
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # 如果每个跟踪框只与一个检测框IOU大于阈值，每个检测框只与一个跟踪框IOU大于阈值，则认为跟踪唯一，大于阈值的跟踪框都是正确的
        # a 是一个（检测对象数量 * 追踪器数量）的矩阵---->有1可以理解为（检测对象和追踪器）匹配到了
        if a.sum(1).max() == 1 and a.sum(0).max() == 1: #有且仅有一个对象而且被匹配到了
            matched_indices = np.stack(np.where(a), axis=1)
        # 否则需要使用线性任务指派算法，将每个检测框与跟踪框以最小代价匹配起来，这也是为什么 iou_matrix 需要乘以 -1 的原因
        else: #存在多个对象而且被追踪到了
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    
    # -----step3 未匹配的 新检测目标&追踪器  ---------
    unmatched_detections = []#未匹配的新检测目标
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = [] #未匹配的追踪器
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # 滤波器输出与低IOU匹配 filter out matched with low IOU
    # 这里之所以还需要筛选一遍，是因为线性任务分配算法是根据全局考虑，有可能某一些分数很差的也被匹配了
    
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])#第0维代表 检测对象对象
            unmatched_trackers.append(m[1])#第1维代表追踪对象
        else:
            matches.append(m.reshape(1, 2))#匹配上的
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)# 去掉其他信息

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age # 最大周期值（未被检测更新的跟踪器随帧数增加），超过之后会被删除
        self.min_hits = min_hits # 目标命中的最小次数，小于该次数不返回
        self.iou_threshold = iou_threshold
        self.trackers = [] # KalmanBoxTracker类型
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
          一个numpy检测数组，格式为[[x1，y1，x2，y2，score]，[x1、y1、x2、y2，score]，…]
        要求：即使检测为空，也必须为每个帧调用一次此方法（对于没有检测的帧，使用np.empty（（0，5）））。
        返回一个类似的数组，其中最后一列是对象ID。
        NOTE: The number of objects returned may differ from the number of detections provided.
        注意：返回的对象数可能与提供的检测数不同。

        """
        self.frame_count += 1 # 帧计数器
        #----------step1 从现有跟踪器获取预测位置（卡尔曼滤波预测） get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5)) # 根据当前卡尔曼跟踪器个数创建， 5 代表 x1,x2,y1,y2,confidence ,从main的dets = seq_dets[seq_dets[:, 0]==frame, 2:7]传入
        to_del = []#想要删除的文件
        ret = [] #最终会输出的
        # step1: predict
        for t, trk in enumerate(trks): #遍历现存的每一个追踪器使用卡尔曼滤波预测位置
            pos = self.trackers[t].predict()[0] # 预测当前物体当前帧的bbox
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0] 
            if np.any(np.isnan(pos)): # 如果预测的bbox为空，删除第t个卡尔曼跟踪器
                to_del.append(t)
        # 删除预测为空跟踪器所在行 trks中存放的是上一帧中被跟踪的所有物体在当前帧预测非空的bbox
         # numpy.ma.masked_invalid 屏蔽出现无效值的数组（NaN 或 inf）
        # numpy.ma.compress_rows 压缩包含掩码值的2-D 数组的整行，将包含掩码值的整行去除
        # trks中存储了上一帧中跟踪的目标并且在当前帧中的预测跟踪框

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) 
        for t in reversed(to_del):# 从跟踪器中删除to_del中的上一帧跟踪器ID
            self.trackers.pop(t)
        # 将目标检测框与卡尔曼滤波器预测的跟踪框关联获取跟踪成功的目标，新增的目标，离开画面的目标 if detect or track failed
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        #----------step2 用指定的检测器更新匹配的跟踪器----- update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :]) #卡尔曼滤波更新

        # create and initialise new trackers for unmatched detections
        # 对于新增的未匹配的检测结果 创建初始化跟踪器trk 并传入trackers
        #----------step3创建并初始化新的跟踪器以进行不匹配的检测------------
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers): # 倒序遍历新的卡尔曼跟踪器集合
            bbox, is_throw = trk.get_state() # 获得追踪器当前状态，box 和是否为抛物行为
            if is_throw and (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((bbox, [trk.id + 1])).reshape(1, -1))  # +1 因为MOT（多目标跟踪）基准要求积为正 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    image = np.zeros([1920, 1080, 3])
    cv2.namedWindow("temp", cv2.WINDOW_NORMAL)
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if (display):
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)  # create instance of the SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split('/')[0]

        with open('output/%s.txt' % (seq), 'w') as out_file:
            print("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:6]
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if (display):
                    fn = 'mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame)
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets)

                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
                    if (display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                        ec=colours[d[4] % 32, :]))

                if (display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    if (display):
        print("Note: to get real runtime results run without the option: --display")
