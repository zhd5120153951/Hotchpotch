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

##8000张图--car,person
import os
from pathlib import Path
import pandas as pd
import cv2

alphabet = ['car', 'person']

label_root = Path("D://FilePackage//datasets//train//labels")  # 替换为实际的标注文件夹路径
image_root = Path("D://FilePackage//datasets//train//images")  # 替换为实际的图像文件夹路径
output_root = Path("D://FilePackage//datasets//train//yolo2img")  # 替换为实际的输出文件夹路径


def paint(label_file, image_file, output_file):
    try:
        # 读取标签
        df = pd.read_csv(label_file, sep=" ", names=['id', 'center-x', 'center-y', 'w', 'h'])
        df['id'] = df['id'].apply(lambda x: alphabet[x])
        df = df.sort_values(by='center-x')
        # 读取图片
        img = cv2.imread(str(image_file))
        h, w = img.shape[:2]

        df[['center-x', 'w']] = df[['center-x', 'w']].apply(lambda x: x * w)
        df[['center-y', 'h']] = df[['center-y', 'h']].apply(lambda x: x * h)

        df['x1'] = df['center-x'] - df['w'] / 2
        df['x2'] = df['center-x'] + df['w'] / 2
        df['y1'] = df['center-y'] - df['h'] / 2
        df['y2'] = df['center-y'] + df['h'] / 2

        df[['x1', 'x2', 'y1', 'y2']] = df[['x1', 'x2', 'y1', 'y2']].astype('int')

        points = zip(df['x1'], df['y1'], df['x2'], df['y2'], df['id'])
        for point in points:
            x1, y1, x2, y2, label = point
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        cv2.imwrite(str(output_file), img)
        print('Generated:', output_file)
    except Exception as e:
        print('Error processing:', label_file)
        print('Error message:', str(e))


# 创建保存新图像的文件夹
output_image_folder = output_root / 'images'

output_image_folder.mkdir(parents=True, exist_ok=True)

# 遍历标注文件夹中的所有txt文件
for label_file in label_root.glob("*.txt"):
    image_file = image_root / (label_file.stem + ".jpg")
    output_image_file = output_image_folder / (label_file.stem + ".jpg")
    paint(label_file, image_file, output_image_file)