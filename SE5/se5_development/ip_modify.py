'''
@FileName   :ip_modify.py
@Description:适用于Ubuntu系统--算能SE5盒子已测试
@Date       :2023/10/24 15:42:17
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import subprocess

new_ip = "192.168.21.30"
mask_code = "255.255.255.0"
gateway = "192.168.21.1"
dns = "183.221.253.100"
#shell 命令--参数可来自前端--根据每个盒子自己的不同来测试
command = "bm_set_ip eth0 {} {} {} {}".format(new_ip, mask_code, gateway, dns)

try:
    # 使用subprocess运行命令
    subprocess.run(command, shell=True, check=True)
    print("命令执行成功")
except subprocess.CalledProcessError as e:
    print(f"命令执行失败: {e}")
