'''
@FileName   :ip_modify.py
@Description:算能盒子SE5通过python脚本修改IP，也可以作为web页面调用来做交互修改自定义IP
@Date       :2022/11/13 10:01:53
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
@PS         :
'''
import subprocess
new_ip = "192.168.21.30"
mask_code = "255.255.255.0"
gateway = "192.168.21.1"
dns = "183.221.253.100"
# shell 命令--参数可来自前端
command = "bm_set_ip eth0 {} {} {} {}".format(new_ip, mask_code, gateway, dns)

try:
    # 使用subprocess运行命令
    subprocess.run(command, shell=True, check=True)
    print("命令执行成功")
except subprocess.CalledProcessError as e:
    print(f"命令执行失败: {e}")
