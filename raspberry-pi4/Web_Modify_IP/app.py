'''
@FileName   :app.py
@Description:
@Date       :2023/11/13 11:15:28
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

from flask import Flask, render_template, request, redirect, url_for
import netifaces
import logging


# 配置基本日志设置
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别，可以选择DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志消息的格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 设置日期时间格式
    filename='./App.log',  # 指定日志输出到文件
    filemode='a'  # 指定文件写入模式（a表示追加，w表示覆盖）
)

# 创建一个日志记录器
logger = logging.getLogger('my_logger')

app = Flask(__name__)


def get_ip_address(interface='eth0'):
    try:
        return netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']
    except KeyError:
        return None


def set_ip_address(interface='eth0', new_ip=None):
    if new_ip:
        try:
            netifaces.ifaddresses(interface)[
                netifaces.AF_INET][0]['addr'] = new_ip
            return True

        except KeyError:
            return False
    else:
        return False


@app.route('./')
def index():
    interface = 'eth0'
    current_ip = get_ip_address(interface)
    return render_template('index.html', interface=interface, current_ip=current_ip)


@app.route('./set_ip', method=['POST'])
def set_ip():
    new_ip = request.form.get('new_ip')
    interface = 'eth0'

    if set_ip_address(interface, new_ip):
        return redirect(url_for('index'))
    else:
        return 'Failed to set IP address...'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
