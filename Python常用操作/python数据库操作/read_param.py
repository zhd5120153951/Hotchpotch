import psycopg2
from torch import le

# 假设你有一个User类


class User:
    def __init__(self, id, name, age, grade):
        self.id = id
        self.name = name
        self.age = age
        self.grade = grade


# 数据库连接参数
conn_params = {
    "host": "192.168.20.5",
    "database": "daito",
    "user": "admin",
    "password": "greatech"
}

# 连接到数据库
try:
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()

    # 执行SQL查询
    # 假设你有一个名为'users'的表
    cur.execute("SELECT id,name,age,grade FROM users")

    # 读取查询结果并创建User类的实例
    users = []

    for row in cur:
        user_id, user_name, user_age, user_grade = row
        user = User(user_id, user_name, user_age, user_grade)
        users.append(user)
    print(f"users num:{len(users)}")
    # 处理users列表...
    for user in users:
        print(
            f"ID: {user.id}, Name: {user.name}, Age: {user.age},Grade:{user.grade}")

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
finally:
    # 关闭数据库连接
    if (conn):
        cur.close()
        conn.close()
        print("PostgreSQL connection is closed")
