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
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Nullable, create_engine
from urllib import parse
from sqlalchemy import text
from flask_migrate import Migrate

app = Flask(__name__)

# app.config()设置好连接信息
# 然后使用SQLALCHMEY()创建一个数据库对象,并自动读取数据库配置信息
HOSTNAME = "127.0.0.1"
PORT = "3306"
USERNAME = "root"
PASSWORD = "ZHDzhd2231"
DATABASE = "test"

pwd = parse.quote_plus(PASSWORD)

app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{USERNAME}:{pwd}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"
# create_engine(
#     f"mysql+pymysql://{USERNAME}:{pwd}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4")


db = SQLAlchemy(app)

migrate = Migrate(app, db)
# 参考知了传课--21集
# flask-migrate:ORM模型映射三部曲:比起通过定义类来创建表要快速
# 1.flask db init:这里只需要执行一次
# 2.flask db migrate:识别ORM模型得改变,生成迁移脚本
# 3.flask db upgrade:运行迁移脚本,同步到数据库中

# 后端向前端html传递一个对象


class User():
    def __init__(self, username, email) -> None:
        self.username = username
        self.email = email

# 通过ORM映射来创建数据库表--create_all()--后面不推荐使用,而是使用Migrate

# 用户-用于创建用户表


class TestUser(db.Model):
    __tablename = "test_user"  # 表名
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# 外键和表的关系测试

# 文章--用于创建文章表


class Article(db.Model):
    __tablename = "article"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)

    # 添加作者外键testuser
    author_id = db.Column(db.Integer, db.ForeignKey("test_user.id"))
    author = db.relationship("TestUser", backref="articles")  # 适合单人开发


# article = Article(title="Flask学习大纲", content="Flask牛逼克拉斯")
# article.user_id=User().id
# user = TestUser.query.get(article.user_id)
# print(article.user)


# 后面有空了详细去了解app_context()
# with app.app_context():
    # db.create_all()#对标ORM
    # with db.engine.connect() as conn:
    #     rs = conn.execute(text("SELECT 1"))
    #     print(rs.fetchone())


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
# 加载静态文件--css,js,img


@app.route("/static")
def static_load():
    return render_template("app4/static.html")

# ORM操纵数据库新增数据--用于注册用户,往数据库中加图等


@app.route("/user/add")
def add_user():
    # 1.创建ORM对象
    user = TestUser(username="法外狂徒张三", password="123456")
    # 2.ORM对象添加到db.session中
    db.session.add(user)
    # 3.把db.session同步到数据库
    db.session.commit()
    return "用户船舰成功"

# ORM操作数据库查询数据库


@app.route("/user/query")
def query_user():
    # 1.get查找,根据主键查找--只查1条---用的少
    user = TestUser.query.get(1)
    print(f"{user.id}:{user.username}--{user.password}")

    # 2.filter_by查找--查多条--用的多
    # users--返回值是一个类类型的数组--可以做切片[]操作
    users = TestUser.query.filter_by(username="法外狂徒张三")
    for user in users:
        print(user.username)
    return "查询成功"

# ORM更行数据库


@app.route("/user/update")
def update_user():
    # 1.获取数据
    user = TestUser.query.filter_by(
        username="法外狂徒张三").first()  # first()代表返回的是一个类对象,不是一个数组
    user.password = "654321"
    db.session.commit()

    return "更新成功"

# ORM删除数据表内容


@app.route("/user/delete")
def delete_user():
    # 1.查找数据
    user = TestUser.query.filter_by(
        username="法外狂徒张三").first()  # first()代表返回的是一个类对象,不是一个数组
    # 2.删除数据记录--session
    db.session.delete(user)
    # 3.同步到数据库
    db.session.commit()

    return "删除成功"

# 外键和表的关系:不同表的关联:一对一,多对一,一对多,多对多


@app.route("/article/add")
def article_add():
    article1 = Article(title="Flask学习大纲", content="地方的风格")
    article1.author = TestUser.query.get(1)

    article2 = Article(title="Djandsd学习大纲", content="grgfef的风格")
    article2.author = TestUser.query.get(1)

    db.session.add_all([article1, article2])
    db.session.commit()

    return "文章创建成功"

# 查找外键和表-区别前面的查找


@app.route("/article/query")
def query_article():
    user = TestUser.query.get(1)
    for article in user.articles:
        print(article.title)
        print(article.content)
    return "查找成功"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
