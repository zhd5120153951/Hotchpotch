'''
@FileName   :子进程2.py
@Description:这个方式优于子进程.py那种方式
@Date       :2022/10/12 14:18:30
@Author     :这个脚本实现主进程--通过while循环不停的重启动子进程(崩溃后:除0操作,被系统杀死)
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''


import time
import multiprocessing as mp


def child_process_func():
    count = 10
    while True:
        print("Child Process is running...")
        time.sleep(1)
        # 模拟崩溃---实际可用try catch捕捉
        # raise Exception("Child process crashed!")
        try:
            print("结果:{}".format(float(10 / count)))
            count -= 2
        except:
            print("except")
            raise Exception("crashed...")


def daemon_process_fun():
    child = mp.Process(target=child_process_func)  # 这个是子进程中的子进程1
    child.start()
    while True:
        # print("Daemon process is running...")
        # 检查子进程状态
        if child.is_alive():
            time.sleep(0.5)
        else:
            print("Child process is crashed*********")
            child.terminate()
            child.join()
            # 创建子进程
            child = mp.Process(target=child_process_func)  # 这个是子进程中的子进程1
            # 启动子进程
            child.start()


# 如果守护进程崩了,就只有让系统服务把主进程拉起,在创建一个新的守护进程了
if __name__ == "__main__":
    # 方式一:主进程中创建子进程--由循环拉起崩溃的子进程

    # 创建子进程
    # child = mp.Process(target=child_process_func)  # 这个是子进程中的子进程1
    # 启动子进程
    # child.start()
    # while True:
    #     if child.is_alive():
    #         time.sleep(0.5)
    #     else:
    #         child.terminate()
    #         child.join()
    #         # 创建子进程
    #         child = mp.Process(target=child_process_func)  # 这个是子进程中的子进程1
    #         # 启动子进程
    #         child.start()

    # 方式二:主进程中创建守护进程,再在守护进程中创建子进程,并在崩溃后由守护进程拉起

    # 创建守护进程
    daemon = mp.Process(target=daemon_process_fun)  # 这个是子进程中的子进程2
    # 启动守护进程
    daemon.start()
    # daemon.join()
    while daemon.is_alive():
        print("dameon process is running---------")
        time.sleep(1)
