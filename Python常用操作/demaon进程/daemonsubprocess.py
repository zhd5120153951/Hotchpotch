import subprocess
import time


def start_daemon_process():
    while True:

        process = subprocess.Popen(['python', '守护进程拉起子进程.py'])

        while True:
            if process.poll() is not None:
                time.sleep(5)
                break
            time.sleep(1)


if __name__ == '__main__':
    start_daemon_process()