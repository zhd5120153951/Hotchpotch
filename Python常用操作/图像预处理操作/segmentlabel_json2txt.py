'''
@FileName   :segmentlabel_json2txt.py
@Description:
@Date       :2024/09/20 13:57:36
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
from ast import arg, parse
import json
import random
import argparse
from sympy import true
import yaml
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# 设置随机种子
random.seed(114514)
image_formats = ['jpg', 'jpeg', 'png', 'tif',
                 'webp', 'bmp', 'dng', 'mpo', 'pfm']


def copy_labeled_img(json_path: Path, target_folder: Path, task: str):
    # 遍历支持的图像格式
    for format in image_formats:
        image_path = json_path.with_suffix('.'+format)
        if image_path.exists():
            # 构建目标路径
            target_path = target_folder/'images'/task/image_path.name
            shutil.copy(image_path, target_path)


def json_to_yolo(json_path: Path, sorted_key: list):
    with open(json_path, 'r') as f:
        labeled_data = json.load(f)
    width = labeled_data['imageWidth']
    height = labeled_data['imageHeight']
    yolo_lines = []
    for shape in labeled_data['shapes']:
        label = shape['label']
        points = shape['points']
        class_ids = sorted_key.index(label)  # 类别下标索引
        txt_str = f'{class_ids} '

        for x, y in points:
            x /= width
            y /= height
            txt_str += f'{x} {y} '
        yolo_lines.append(txt_str.strip()+'\n')
    return yolo_lines


def create_directory(directory_folder):
    directory_folder.mkdir(parents=True, exist_ok=True)
# 创建使用的yaml文件


def create_yaml(output_folder: Path, sorted_key: list):
    train_img_path = Path('images')/'train'
    val_img_path = Path('images')/'val'
    train_label_path = Path('labels')/'train'
    val_label_path = Path('labels')/'val'

    # 创建目录
    for path in [train_img_path, val_img_path, train_label_path, val_label_path]:
        create_directory(output_folder / path)
    names_dict = {idx: name for idx, name in enumerate(sorted_key)}
    yaml_dict = {
        'path': output_folder.as_posix(),
        'train': train_img_path.as_posix(),
        'val': val_img_path.as_posix(),
        'names': names_dict
    }
    yaml_file_path = output_folder/'yolo.yaml'
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_dict, yaml_file,
                  default_flow_style=False, sort_keys=False)
    print(f'yaml file created in {yaml_file_path.as_posix()}')


def get_labels_and_json_path(input_folder: Path):
    json_file_paths = list(input_folder.rglob('*.json'))
    label_counts = defaultdict(int)
    for json_file_path in json_file_paths:
        with open(json_file_path, 'r') as json_file:
            label_data = json.load(json_file)
        for shape in label_data['shapes']:
            label = shape['label']
            label_counts[label] += 1
    # 根据标签出现次数排序标签
    sorted_keys = sorted(
        label_counts, key=lambda k: label_counts[k], reverse=True)
    return sorted_keys, json_file_paths


def label_to_yolo(json_file_paths: list, output_folder: Path, sorted_key: list, split_rate: float):
    # 随机打乱 JSON 文件路径列表
    random.shuffle(json_file_paths)

    # 计算训练集和验证集的分割点
    split_point = int(split_rate * len(json_file_paths))
    train_set = json_file_paths[:split_point]
    val_set = json_file_paths[split_point:]

    for json_file_path in tqdm(train_set):
        txt_name = json_file_path.with_suffix(".txt").name
        yolo_lines = json_to_yolo(json_file_path, sorted_key)
        output_json_path = Path(output_folder / "labels" / "train" / txt_name)
        with open(output_json_path, "w") as f:
            f.writelines(yolo_lines)
        copy_labeled_img(json_file_path, output_folder, task="train")

    for json_file_path in tqdm(val_set):
        txt_name = json_file_path.with_suffix(".txt").name
        yolo_lines = json_to_yolo(json_file_path, sorted_key)
        output_json_path = Path(output_folder / "labels" / "val" / txt_name)
        with open(output_json_path, "w") as f:
            f.writelines(yolo_lines)
        copy_labeled_img(json_file_path, output_folder, task="val")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='label2yolo')
    parser.add_argument('input_folder', default='E:\\Datasets\\belt\\belt_seg_v8',
                        help='input labeled files folder')
    parser.add_argument(
        'output_folder', default='E:\\Datasets\\belt\\belt_seg_v8', help='output txt files folder')
    parser.add_argument('split_rate', default='0.2', help='train and val rate')

    args = parser.parse_args()
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    split_rate = float(args.split_rate)

    sorted_key, json_file_paths = get_labels_and_json_path(input_folder)
    create_yaml(output_folder, sorted_key)
    label_to_yolo(json_file_paths, output_folder, sorted_key, split_rate)
