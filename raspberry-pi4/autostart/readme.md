# 一、树莓派开机自启动python(服务的方式)

**以我的树莓派为例**

1、可直接在当前目录下执行:
    `sudo vim /etc/systemd/system/autopython.service`
2、输入服务内容：

    [Unit]
    Description=My Auto Python Script

    [Service]
    ExecStart=/usr/bin/python /home/daito/Documents/python_project/yolov5-lite/test.py
    WorkingDirectory=/home/daito/Documents/python_project/yolov5-lite/
    Restart=always(always表示进程崩溃后，服务又会自动拉起来)
    User=daito

    [Install]
    WantedBy=multi-user.target
    
    保存并退出
3、检查一下要执行的脚本(py)的权限，没有则给其权限:
    `chmod +x /home/daito/Documents/python_project/yolov5-lite/*(test.py) `
4、启动systemd服务:
    `sudo systemctl enable autopython.service`
    `sudo systemctl start autopython.service`
5、如果出错，可查看日志：
    `sudo journalctl -u autopython.service`

到此自启动的python服务就开启并开始了。

**一般而言：开启自启动的服务都是为了长期、永久的运行某个任务(进程)开启后一般不会停止(stop)，甚至关闭(disable)**

# 二、如果想要停止这项服务，甚至关闭服务，如下：

1、停止服务：(仅对当此生效，设备下次重启，还是会自启动并一直执行)
    `sudo systemctl stop autopython.service`
2、关闭服务：(永久关停服务，除非再次开启)
    `sudo systemctl disable autopython.service`

# 三、如果python程序是由某个守护进程拉起,不是通过服务的方式(崩溃后守护进程拉起)，怎么关掉该进程(临时因为守护进程在，除非杀掉守护)
1、pkill命令或者kill命令
2、`pgrep -f 被守护的python.py` 或者 `ps aux | grep 被守护的python.py`
3、`sudo kill/pkill 进程号(PID)`

# 四、服务的方式和守护进程启动python的区别？

*使用服务方式启动Python和使用守护进程方式启动Python有一些重要区别，它们适用于不同的使用场景和需求。下面是关于这两种方式的主要区别：*

1. 启动时机：

服务方式： 服务通常在系统启动时或特定事件发生时启动。这意味着你可以配置服务在系统启动时自动启动，或者在需要时手动启动。

守护进程方式： 守护进程是在需要时手动启动的，通常由管理员或用户触发。它们不会在系统启动时自动启动。

2. 进程控制：

服务方式： 服务通常由系统管理工具（如systemd）负责启动、停止和监控。你可以使用systemctl等命令来管理服务，包括启动、停止、重启、查看日志等。

守护进程方式： 守护进程通常由命令行或脚本手动启动，并且你需要手动停止它们。它们通常不由系统管理工具进行监控。

3. 监控和自恢复：

服务方式： 服务可以配置为在崩溃或失败时自动重启。这使得它们更稳定，因为系统会自动尝试恢复。

守护进程方式： 守护进程通常需要额外的逻辑来处理崩溃和重启。你需要编写脚本或使用其他工具来监控它们，并在需要时重新启动它们。

4. 管理和维护：

服务方式： 服务的管理通常更容易，因为系统管理工具提供了方便的命令和日志记录。服务通常更适合在后台运行并提供系统级服务，如Web服务器、数据库服务器等。

守护进程方式： 守护进程的管理可能需要更多的手动工作，包括编写自定义脚本来处理进程的启动、停止和监控。它们通常更适合用于特定用途的自定义脚本或工具。

*综上所述，选择使用服务方式或守护进程方式启动Python取决于你的需求和场景。如果你需要在系统启动时自动启动并希望系统能够自动管理进程的生命周期，服务方式更合适。如果你需要手动启动并管理Python进程，守护进程方式可能更适合。*