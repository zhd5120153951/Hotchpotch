'''
@FileName   :segmentlabel_txt2json.py
@Description:
@Date       :2025/05/12 11:29:03
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
from pathlib import Path
import yaml
import argparse
import json
import numpy as np

"""
txt_to_json.py

将 YOLOv5-seg/v8-seg 格式的 txt 标签（polygon）转换为 LabelMe JSON 格式。

用法：
    python txt_to_json.py  dataset.yaml labels images labels_json

参数说明：
    --data_yaml   包含 'names' 字段的 data.yaml 文件，用以获取类别名映射
    --labels_dir  存放 txt 标签的根目录（包含 train/ val/ 子目录）
    --images_dir  存放原始图像的根目录（包含 train/ val/ 子目录）
    --output_dir  输出 JSON 的根目录（将生成 train/ val/ 下的 .json 文件）
"""


def load_class_names(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    # names: {0: 'cat', 1: 'dog', ...}
    names_dict = data.get('names', {})
    # convert to list by sorted index
    return [names_dict[i] for i in sorted(names_dict.keys(), key=int)]


def txt_to_json(txt_path: Path, img_path: Path, class_names: list) -> dict:
    """
    将单个 txt 文件（polygon 格式）转换为 JSON dict。
    """
    # 1. 读取图片尺寸
    # img = cv2.imread(str(img_path))
    img = cv2.imdecode(np.fromfile(
        img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    h, w = img.shape[:2]

    # 2. 解析 txt 内容
    shapes = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7 or (len(parts)-1) % 2 != 0:
                # 每行至少要 class_id + 3 个点 (6 coords)
                continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            # 恢复为像素坐标
            points = []
            for i in range(0, len(coords), 2):
                x_norm, y_norm = coords[i], coords[i+1]
                x = x_norm * w
                y = y_norm * h
                points.append([x, y])
            shapes.append({
                "label": class_names[cls_id],
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

    # 3. 构造 JSON 主体
    json_dict = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }
    return json_dict


def convert_dir(labels_dir: Path, images_dir: Path, output_dir: Path, class_names: list):
    """
    遍历 labels_dir 下的 train/ val/ 子目录，将每个 txt 转为 json。
    """
    for split in ["train", "val"]:
        txt_folder = labels_dir / split
        img_folder = images_dir / split
        out_folder = output_dir / split
        out_folder.mkdir(parents=True, exist_ok=True)

        for txt_path in txt_folder.glob("*.txt"):
            img_path = img_folder / txt_path.with_suffix('').name
            # 寻找支持的图像后缀
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp']:
                candidate = img_path.with_suffix(ext)
                if candidate.exists():
                    img_path = candidate
                    break
            else:
                print(f"[警告] 未找到对应图像: {txt_path.stem}.*")
                continue

            # 转换
            json_dict = txt_to_json(txt_path, img_path, class_names)
            out_file = out_folder / (txt_path.stem + ".json")
            with open(out_file, 'w', encoding='utf-8') as jf:
                json.dump(json_dict, jf, ensure_ascii=False, indent=2)
            print(f"已生成: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-Seg txt → LabelMe JSON 转换")
    parser.add_argument("data_yaml", default="",
                        help="包含 names 字段的 data.yaml")
    parser.add_argument("labels_dir", default="",
                        help="txt 标签根目录（含 train/ val）")
    parser.add_argument("images_dir", default="",
                        help="图像根目录（含 train/ val）")
    parser.add_argument("output_dir", default="",
                        help="输出 JSON 根目录")
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    labels_dir = Path(args.labels_dir)
    print(f"标签目录: {labels_dir}")
    images_dir = Path(args.images_dir)
    print(f"图像目录: {images_dir}")
    output_dir = Path(args.output_dir)
    print(f"输出目录: {output_dir}")
    class_names = load_class_names(data_yaml)
    convert_dir(labels_dir, images_dir, output_dir, class_names)


if __name__ == "__main__":
    main()
