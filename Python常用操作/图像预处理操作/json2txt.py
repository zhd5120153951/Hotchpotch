'''
@FileName   :json2txt.py
@Description:json转化txt格式--yolo格式
@Date       :2026/01/23 16:15:00
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import json
import os
import cv2

name2id = {'henggang': 0, 'luoshuan': 1}


def convert(img_size, box):
    # img_size: (width, height)
    width, height = img_size
    if width == 0 or height == 0:
        return (0, 0, 0, 0)
    dw = 1.0 / width
    dh = 1.0 / height
    x_min, y_min, x_max, y_max = box
    x = (x_min + x_max) / 2.0
    y = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return (x * dw, y * dh, w * dw, h * dh)


def decode_json(json_floder_path, txt_outer_path, json_name, jpgs_path):
    # 只处理 .json 文件
    if not json_name.lower().endswith('.json'):
        return

    txt_name = os.path.join(
        txt_outer_path, os.path.splitext(json_name)[0] + '.txt')
    json_path = os.path.join(json_floder_path, json_name)

    try:
        with open(json_path, 'r', encoding='utf-8', errors='ignore') as jf:
            data = json.load(jf)
    except Exception as e:
        print('无法读取 json:', json_path, e)
        return

    img_w = data.get('imageWidth', 0)
    img_h = data.get('imageHeight', 0)
    imgpath = data.get('imagePath', '')

    # 尝试读取图片以获取尺寸（若 json 中没有或为 0）
    img = None
    if imgpath:
        if os.path.isabs(imgpath):
            img_full = imgpath
        else:
            img_full = os.path.join(jpgs_path, imgpath)
        if os.path.exists(img_full):
            img = cv2.imread(img_full)
        else:
            # 有时 imagePath 只是文件名
            alt = os.path.join(jpgs_path, os.path.basename(imgpath))
            if os.path.exists(alt):
                img = cv2.imread(alt)

    if img is not None:
        h, w = img.shape[:2]
        if not img_h:
            img_h = h
        if not img_w:
            img_w = w
    else:
        if not img_w or not img_h:
            print('警告: 无法读取图片且 json 中尺寸为空，跳过：', json_name)
            return

    shapes = data.get('shapes', [])
    if not shapes:
        # 写空文件（或不写）
        open(txt_name, 'w').close()
        return

    os.makedirs(txt_outer_path, exist_ok=True)
    with open(txt_name, 'w', encoding='utf-8') as f:
        for s in shapes:
            label_name = s.get('label', '')
            shape_type = s.get('shape_type', '').lower()
            points = s.get('points', [])

            if not points:
                continue

            # 统一计算 x_min,y_min,x_max,y_max
            try:
                xs = [float(p[0]) for p in points]
                ys = [float(p[1]) for p in points]
                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)
            except Exception:
                continue

            bbox = (x_min, y_min, x_max, y_max)
            yolo_box = convert((img_w, img_h), bbox)

            if label_name not in name2id:
                print('未知标签，跳过:', label_name, 'in', json_name)
                continue

            cls_id = name2id[label_name]
            f.write('{} {}'.format(cls_id, ' '.join(
                [str(a) for a in yolo_box])) + '\n')


if __name__ == '__main__':
    # 用户请根据本机实际路径修改以下三个路径（保持一致的 Dataset(s) 名称）
    jpgs_path = 'E:\\Datasets\\slny_gbj1'  # 存放图片的文件夹的绝对路径
    json_floder_path = 'E:\\Datasets\\slny_gbj1'  # 存放 json 的文件夹的绝对路径
    txt_outer_path = 'E:\\Datasets\\slny_gbj1'  # 存放 txt 的文件夹绝对路径

    if not os.path.exists(json_floder_path):
        print('json 文件夹不存在：', json_floder_path)
    else:
        json_names = sorted(os.listdir(json_floder_path))
        print('共有：{} 个文件待转化'.format(len(json_names)))
        flagcount = 0
        for json_name in json_names:
            decode_json(json_floder_path, txt_outer_path, json_name, jpgs_path)
            flagcount += 1
            print('还剩下{}个文件未转化'.format(len(json_names) - flagcount))

        print('转化全部完毕')
