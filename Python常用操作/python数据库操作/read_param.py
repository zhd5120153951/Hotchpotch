import psycopg2
# import pyyaml

# User类


class User:
    def __init__(self, id, name, age, grade):
        self.id = id
        self.name = name
        self.age = age
        self.grade = grade


class AI_ALGORITHM(object):
    def __init__(self, id, algorithm_name, algorithm_code, algorithm_type, algorithm_desc, algorithm_default_config, deleted, creator, updated_by, create_date, update_date) -> None:
        self.id = id
        self.name = algorithm_name
        self.code = algorithm_code
        self.type = algorithm_type
        self.desc = algorithm_desc
        self.default_config = algorithm_default_config
        self.deleted = deleted
        self.creator = creator
        self.updated_by = updated_by
        self.create_date = create_date
        self.update_date = update_date


class AI_CONFIG_PUSH_RECORD(object):
    def __init__(self, id, node_id, remark, push_time, end_time, push_status, deleted, creator, updated_by, create_date, update_date) -> None:
        self.id = id
        self.node_id = node_id
        self.remark = remark
        self.push_time = push_time
        self.end_time = end_time
        self.push_status = push_status
        self.deleted = deleted
        self.creator = creator
        self.updated_by = updated_by
        self.create_date = create_date
        self.update_date = update_date


class AI_NODE(object):
    def __init__(self, id, node_code, node_name, node_ip, status, deleted, creator, updated_by, create_date, update_date, node_port, last_heart_beat_time, error, error_msg) -> None:
        self.id = id
        self.node_code = node_code
        self.node_name = node_name
        self.node_ip = node_ip
        self.status = status
        self.deleted = deleted
        self.creator = creator
        self.updated_by = updated_by
        self.create_date = create_date
        self.update_date = update_date
        self.node_port = node_port
        self.last_heart_beat_time = last_heart_beat_time
        self.error = error
        self.error_msg = error_msg


class CAMERA_BINDING_ALGORITHM(object):
    def __init__(self, id, resource_id, algorithm_id, algorithm_config, region_config, enable, alarm_level) -> None:
        self.id = id
        self.resource_id = resource_id
        self.algorithm_id = algorithm_id
        self.algorithm_config = algorithm_config
        self.region_config = region_config
        self.enable = enable
        self.alarm_level = alarm_level


class CAMERA_BINDING_NODE(object):
    def __init__(self, id, resource_id, node_id) -> None:
        self.id = id
        self.resource_id-resource_id
        self.node_id = node_id


class DC_DICT(object):
    def __init__(self, id, code, en_name, cn_name, remark, deleted, creator_id, create_time, last_update_time) -> None:
        self.id = id
        self.code = code
        self.en_name = en_name
        self.cn_name = cn_name
        self.remark = remark
        self.deleted = deleted
        self.creator_id = creator_id
        self.create_time = create_time
        self.last_update_time = last_update_time


class DC_DICT_ITEM(object):
    def __init__(self, id, dict_code, cn_name, value, en_name, sort, status, defaulted, remark, icon, deleted, creator_id, create_time, last_update_time) -> None:
        self.id = id
        self.dict_code = dict_code
        self.cn_name = cn_name
        self.value = value
        self.en_name = en_name
        self.sort = sort
        self.status = status
        self.defaulted = defaulted
        self.remark = remark
        self.icon = icon
        self.deleted = deleted
        self.creator_id = creator_id
        self.create_time = create_time
        self.last_update_time = last_update_time


class VA_WORK_ALARM_RESOURCE(object):
    def __init__(self, id, work_id, resource_id, alarm_type, alarm_level) -> None:
        self.id = id
        self.work_id = work_id
        self.resource_id = resource_id
        self.alarm_type = alarm_type
        self.alarm_level = alarm_level


class VA_WORK_INFO(object):
    def __init__(self, id, work_number, work_type, work_level, work_org_id, work_org_name, work_area_id, work_area_name, work_status, work_start_date_time, work_end_date_time, work_content, work_applicant, work_address, coordinate, work_personnel, work_guardianship, work_approval, work_approval_status, work_part, work_describe, jsa_analysis_results, work_id) -> None:
        self.id = id
        self.work_number = work_number
        self.work_type = work_type
        self.work_level = work_level
        self.work_org_id = work_org_id
        self.work_org_name = work_org_name
        self.work_area_id = work_area_id
        self.work_area_name = work_area_name
        self.work_status = work_status
        self.work_start_date_time = work_start_date_time
        self.work_end_date_time = work_end_date_time
        self.work_content = work_content
        self.work_applicant = work_applicant
        self.work_address = work_address
        self.coordinate = coordinate
        self.work_personnel = work_personnel
        self.work_guardianship = work_guardianship
        self.work_approval = work_approval
        self.work_approval_status = work_approval_status
        self.work_part = work_part
        self.work_describe = work_describe
        self.jsa_analysis_results = jsa_analysis_results
        self.work_id = work_id


# 数据库连接参数
conn_params = {
    "host": "192.168.20.5",
    "database": "postgres",
    "user": "admin",
    "password": "greatech"
}

# # 连接到数据库
# try:
#     conn = psycopg2.connect(**conn_params)
#     cur = conn.cursor()

#     # 执行SQL查询,users的表
#     cur.execute("SELECT id,name,age,grade FROM users")

#     # 读取查询结果并创建User类的实例
#     users = []

#     for row in cur:
#         user_id, user_name, user_age, user_grade = row
#         user = User(user_id, user_name, user_age, user_grade)
#         users.append(user)
#     print(f"users num:{len(users)}")
#     # 处理users列表...
#     for user in users:
#         print(
#             f"ID: {user.id}, Name: {user.name}, Age: {user.age},Grade:{user.grade}")

# except (Exception, psycopg2.Error) as error:
#     print("Error while connecting to PostgreSQL", error)
# finally:
#     # 关闭数据库连接
#     if (conn):
#         cur.close()
#         conn.close()
#         print("PostgreSQL connection is closed")


def conn_pgsql():
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    return conn, cur


def select_pgsql(cur, select_sql):
    cur.execute(select_sql)
    # users = []
    camera_binding_algo = []
    for row in cur:
        # user_id, user_name, user_age, user_grade = row
        # user = User(user_id, user_name, user_age, user_grade)
        # users.append(user)
        id, resource_id, algorithm_id, algorithm_config, region_config, enable, alarm_level = row
        camera_bind_algo = CAMERA_BINDING_ALGORITHM(
            id, resource_id, algorithm_id, algorithm_config, region_config, enable, alarm_level)
        camera_binding_algo.append(camera_bind_algo)
    # return users
    return camera_binding_algo


def close_pgsql(conn, cur):
    if conn:
        cur.close()
        conn.close()
        print("PostgreSQL connection is closed")


if __name__ == "__main__":
    conn, cur = conn_pgsql()
    rets = select_pgsql(
        cur, "SELECT * FROM video_analysis.camera_binding_algorithm WHERE id = '1'")

    print(f"users num:{len(rets)}")
    # 处理users列表...
    for ret in rets:
        print(f"id:{ret.id} resource_id:{ret.resource_id} algorithm_id:{ret.algorithm_id} algorithm_config:{ret.algorithm_config} region_config:{ret.region_config} enable:{ret.enable} alarm_level:{ret.alarm_level}")
    close_pgsql(conn, cur)
