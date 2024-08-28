'''
@FileName   :filter_img_by_size.py
@Description:
@Date       :2024/08/28 14:27:48
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

from utils.filterImage import ImageFilter
import yaml


def read_yaml(yaml_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"{yaml_path}文件找不到...")
    except yaml.YAMLError as yye:
        print(f"YAML解析错误：{yye}")
    except Exception as ex:
        print(f"读取文件时发生错误：{ex}")


if __name__ == "__main__":
    path = read_yaml("./config.yaml")
    src_img_path = path["src_img_path"]
    target_img_path = path["target_img_path"]
    img_filter = ImageFilter(src_img_path, target_img_path)
    img_filter.create_target_path()
    img_filter.filter_move_image()
