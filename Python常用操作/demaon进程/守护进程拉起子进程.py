'''
@FileName   :守护进程拉起子进程.py
@Description:
@Date       :2022/10/12 14:18:30
@Author     :daito
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
        #模拟崩溃---实际可用try catch捕捉
        # raise Exception("Child process crashed!")
        try:
            print("结果:{}".format(float(10 / count)))
            count -= 2
        except:
            print("except")
            raise Exception("crashed...")


def daemon_process_fun():
    global child
    while True:
        print("Daemon process is running...")
        time.sleep(1)
        #检查子进程状态
        if not child.is_alive():
            print("Child process crashed, restarting...")
            #重启子进程
            child.start()
        else:
            print("Child process is running...")
        time.sleep(1)


if __name__ == "__main__":
    # 创建子进程
    child = mp.Process(target=child_process_func)
    #启动子进程
    child.start()

    #创建守护进程
    daemon = mp.Process(target=daemon_process_fun)
    # 设置该进程为守护进程
    daemon.daemon = True
    #启动守护进程
    daemon.start()

    #等待子进程结束
    child.join()