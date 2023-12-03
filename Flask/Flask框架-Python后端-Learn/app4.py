'''
@FileName   :app.py
@Description:
@Date       :2023/11/29 17:00:52
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

from flask import Flask, request, render_template
from datetime import datetime


app = Flask(__name__)


# 后端向前端html传递一个对象


class User():
    def __init__(self, username, email) -> None:
        self.username = username
        self.email = email


# 路由函数(url地址)


@app.route("/")
def index():
    # 得到对象
    user = User(username="daito", email="XX@greatech.com")

    person = {
        "username": "lucy",
        "email": "adad@qq.com"
    }
    return render_template("app2/index.html", user=user, person=person)

# 传参数


@app.route("/blog/<blog_id>")
def blog_list(blog_id):
    return render_template("app2/blog_list.html", blog_id=blog_id)


# 获取当前时间


def get_systime_format(value, format="%Y年%m月%d日 %H:%M:%S"):
    return value.strftime(format)


# 自定义过滤器--获取系统时间
app.add_template_filter(get_systime_format, "dformat")
# Jinja自带过滤器


@app.route("/filter")
def filter():
    user = User("daito-阿东", "zenghedong@outlook.com")
    sysTime = datetime.now()  # 可以一直获取
    return render_template("app3/filter.html", user=user, sysTime=sysTime)

# 在html页面使用if控制传入的参数


@app.route("/control")
def control():
    age = 19
    return render_template("app4/control.html", age=age)

# 模板继承


@app.route("/child1")
def child1():
    return render_template("app4/child1.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
