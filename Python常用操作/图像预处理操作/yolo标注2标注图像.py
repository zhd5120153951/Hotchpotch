'''
@FileName   :yolo标注2标注图像.py
@Description:这段代码实现了一个功能，用于读取标注文件和对应的图像文件，并在图像上绘制标注框和标签，最后将绘制结果保存为新的图像文件。使用时需替换图像路径、标注文件路径和生成图片的保存路径

@Date       :2023/10/07 14:32:54
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import os
from pathlib import Path
import pandas as pd
import cv2

# alphabet3 = ['fire', 'smoke', 'person']
classes = ['pd_kz', 'pd_fz', 'pd_fzs', 'pd_yw', 'pd_dkm']
label_root = Path(
    "E:\\Datasets\\belt\\belt_use\\roboflow_240918_train")  # 替换为实际的标注文件夹路径
image_root = Path(
    "E:\\Datasets\\belt\\belt_use\\roboflow_240918_train")  # 替换为实际的图像文件夹路径
output_root = Path(
    "E:\\Datasets\\belt\\yolo2img")  # 替换为实际的输出文件夹路径


def paint(label_file, image_file, output_file):
    try:
        # 读取标签
        df = pd.read_csv(label_file, sep=" ", names=[
                         'id', 'center-x', 'center-y', 'w', 'h'])
        df['id'] = df['id'].apply(lambda x: classes[x])
        df = df.sort_values(by='center-x')
        # 读取图片
        img = cv2.imread(str(image_file))
        h, w = img.shape[:2]

        # 作用是把df[['center-x','w']]对应的值(txt中)--传到x,并x*w,最后覆盖到df[['center-x','w']]的值
        # 简单说就是把x,y,w,h由比例--数值
        df[['center-x', 'w']] = df[['center-x', 'w']].apply(lambda x: x * w)
        df[['center-y', 'h']] = df[['center-y', 'h']].apply(lambda x: x * h)
        # 画框需要知道左上,右下坐标
        df['x1'] = df['center-x'] - df['w'] / 2
        df['x2'] = df['center-x'] + df['w'] / 2
        df['y1'] = df['center-y'] - df['h'] / 2
        df['y2'] = df['center-y'] + df['h'] / 2

        df[['x1', 'x2', 'y1', 'y2']] = df[[
            'x1', 'x2', 'y1', 'y2']].astype('int')

        points = zip(df['x1'], df['y1'], df['x2'], df['y2'], df['id'])
        for point in points:
            x1, y1, x2, y2, label = point
            img = cv2.rectangle(img, (x1, y1), (x2, y2),
                                color=(0, 255, 0), thickness=1)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        cv2.imwrite(str(output_file), img)
        print('Generated:', output_file)
    except Exception as e:
        print('Error processing:', label_file)
        print('Error message:', str(e))


def cropImg(label_file, image_file, output_image_file, output_txt_file):
    try:
        person_x1, person_y1, person_x2, person_y2 = 0, 0, 0, 0
        phone_x1, phone_y1, phone_x2, phone_y2 = 0, 0, 0, 0
        # 读取标签
        df = pd.read_csv(label_file, sep=" ", names=[
                         'id', 'center-x', 'center-y', 'w', 'h'])

        df['id'] = df['id'].apply(lambda x: classes[x])
        df = df.sort_values(by='center-x')
        # 读取图片
        img = cv2.imread(str(image_file))
        h, w = img.shape[:2]

        # 作用是把df[['center-x','w']]对应的值(txt中)--传到x,并x*w,最后覆盖到df[['center-x','w']]的值
        # 简单说就是把x,y,w,h由比例-->数值
        df[['center-x', 'w']] = df[['center-x', 'w']].apply(lambda x: x * w)
        df[['center-y', 'h']] = df[['center-y', 'h']].apply(lambda x: x * h)
        # 画框需要知道左上,右下坐标
        df['x1'] = df['center-x'] - df['w'] / 2
        df['x2'] = df['center-x'] + df['w'] / 2
        df['y1'] = df['center-y'] - df['h'] / 2
        df['y2'] = df['center-y'] + df['h'] / 2

        df[['x1', 'x2', 'y1', 'y2']] = df[[
            'x1', 'x2', 'y1', 'y2']].astype('int')

        points = zip(df['x1'], df['y1'], df['x2'], df['y2'], df['id'])
        list_points = list(points)
        if len(list_points) != 2:  # 默认一张图中只有一个人和一个电话
            return
        else:
            for point in list_points:
                x1, y1, x2, y2, label = point
                if label == "person":
                    img_person = img[y1:y2, x1:x2]  # 截取的人--需要保存的图
                    cv2.imwrite(str(output_image_file), img_person)
                    person_x1 = x1
                    person_y1 = y1
                    person_x2 = x2
                    person_y2 = y2

                if label == "cigarette":
                    phone_x1 = x1
                    phone_y1 = y1
                    phone_x2 = x2
                    phone_y2 = y2

            phone_w = phone_x2-phone_x1
            phone_h = phone_y2-phone_y1
            person_w = person_x2-person_x1
            person_h = person_y2-person_y1
            phone_center_x_ratio = (phone_x1-person_x1 + phone_w/2)/person_w
            phone_center_y_ratio = (phone_y1-person_y1 + phone_h/2)/person_h
            phone_w_ratio = phone_w/person_w
            phone_h_ratio = phone_h/person_h
            txt_label = f"1 {phone_center_x_ratio} {phone_center_y_ratio} {phone_w_ratio} {phone_h_ratio}"

            # 保存修改后的内容
            with open(output_txt_file, 'w') as file:
                file.writelines(txt_label)
            # img = cv2.rectangle(img, (x1, y1), (x2, y2),color=(0, 255, 0), thickness=1)
            # img = img[y1:y2, x1:x2]
            # cv2.putText(img, label, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        print('Generated:', output_image_file)
    except Exception as e:
        print('Error processing:', label_file)
        print('Error message:', str(e))


# 创建保存新图像的文件夹
output_image_folder = output_root / 'images'
output_txt_folder = output_root / 'txt'

output_image_folder.mkdir(parents=True, exist_ok=True)
output_txt_folder.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    # 遍历标注文件夹中的所有txt文件
    for label_file in label_root.glob("*.txt"):
        image_file = image_root / (label_file.stem + ".jpg")
        output_image_file = output_image_folder / (label_file.stem + ".jpg")
        output_txt_file = output_txt_folder / (label_file.stem+".txt")
        paint(label_file, image_file, output_image_file)
        # cropImg(label_file, image_file, output_image_file, output_txt_file)
