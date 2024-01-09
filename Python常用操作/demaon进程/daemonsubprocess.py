'''
@FileName   :daemonsubprocess.py
@Description:这个脚本可以设为启动服务--让主程序开机自启动,并执行python xxx.py的程序,再守护其他进程
@Date       :2023/12/31 09:57:02
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import subprocess
import time


def start_daemon_process():
    while True:

        process = subprocess.Popen(['python', '子进程.py'])

        while True:
            if process.poll() is not None:
                time.sleep(5)
                break
            time.sleep(1)


if __name__ == '__main__':
    start_daemon_process()
