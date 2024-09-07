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
                parts[0] = '1'
                lines[i] = ' '.join(parts)+'\n'
            else:
                continue
            # if parts[0] == '1':
            #     continue
            # elif parts[1] == '1':
            #     del lines[i - 1]
            # else:
            #     parts[0] = '0'
            #     lines[i] = ' '.join(parts) + '\n'

            # if parts:
            #     parts[0] = '0'  # 用新值替换第一个值
            #     lines[i] = ' '.join(parts) + '\n'  # 重新组合行

        # 保存修改后的内容
        with open(file_path, 'w') as file:
            file.writelines(lines)


def modify_txt2(txt_path):
    txt_names = os.listdir(txt_path)
    for txt_name in txt_names:
        # 读取文本文件
        file_path = os.path.join(txt_path, txt_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 修改第一个值
        for i, line in enumerate(lines):
            parts = line.split()  # 假设文本以空格分隔
            if parts[0] == '0':  # 第一类
                del lines[i]
            elif parts[0] == '1':  # 第二类
                parts[0] = '0'
                lines[i] = ' '.join(parts)+'\n'
            else:
                continue
            # if parts[0] == '1':
            #     continue
            # elif parts[1] == '1':
            #     del lines[i - 1]
            # else:
            #     parts[0] = '0'
            #     lines[i] = ' '.join(parts) + '\n'

            # if parts:
            #     parts[0] = '0'  # 用新值替换第一个值
            #     lines[i] = ' '.join(parts) + '\n'  # 重新组合行

        # 保存修改后的内容
        with open(file_path, 'w') as file:
            file.writelines(lines)


def calcuBox(txt_path):
    cnt = 0
    txt_names = os.listdir(txt_path)
    for txt_name in txt_names:
        # 读取文本文件
        file_path = os.path.join(txt_path, txt_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            cnt = cnt+len(lines)
    return cnt


if __name__ == "__main__":
    txt_path = 'D:\\FilePackage\\datasets\\phonecall\\labeled\\4\\txt'
    # modify_txt2(txt_path)
    count = calcuBox(txt_path)
    print(f"总共有{count}个框...")
