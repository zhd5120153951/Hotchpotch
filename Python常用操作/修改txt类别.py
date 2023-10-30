import os


def modify_txt(txt_path):
    txt_names = os.listdir(txt_path)
    for txt_name in txt_names:
        # 读取文本文件
        file_path = os.path.join(txt_path, txt_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 修改第一个值
        for i, line in enumerate(lines):
            parts = line.split()  # 假设文本以空格分隔
            if parts[0] == '0':
                continue
            elif parts[1] == '1':
                del lines[i - 1]
            else:
                parts[0] = '0'
                lines[i] = ' '.join(parts) + '\n'
            # if parts:
            #     parts[0] = '0'  # 用新值替换第一个值
            #     lines[i] = ' '.join(parts) + '\n'  # 重新组合行

        # 保存修改后的内容
        with open(file_path, 'w') as file:
            file.writelines(lines)


if __name__ == "__main__":
    txt_path = 'D:\\FilePackage\\datasets\\gas2_txt'
    modify_txt(txt_path)
