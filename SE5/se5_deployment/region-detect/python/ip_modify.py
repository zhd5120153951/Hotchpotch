import subprocess
new_ip = "192.168.21.30"
mask_code = "255.255.255.0"
gateway = "192.168.21.1"
dns = "183.221.253.100"
#shell 命令--参数可来自前端
command = "bm_set_ip eth0 {} {} {} {}".format(new_ip,mask_code,gateway,dns)

try:
    # 使用subprocess运行命令
    subprocess.run(command, shell=True, check=True)
    print("命令执行成功")
except subprocess.CalledProcessError as e:
    print(f"命令执行失败: {e}")
