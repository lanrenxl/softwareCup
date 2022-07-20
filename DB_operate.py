# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2022/4/27 18:24
# ------------------------------------------------
import pymssql  # 引入pymssql模块


# 没有权限判断
# 数据库连接
# 数据库增删改查
class db_operator:
    # 连接数据库
    def __init__(self):
        self.my_connect = pymssql.connect(host="127.0.0.1", server='LANRENL', database='007')  # 服务器名,账户,密码,数据库名
        if self.my_connect:
            print("连接数据库成功!\n")
        self.my_cursor = self.my_connect.cursor()
        pass

    # 登录
    # 数据需要按照指定格式传入
    # 没有数据格式的判断
    def user_log_in(self, phoneNum, password):
        sql_select = "select  * from [user] where id='%s'" % phoneNum
        self.my_cursor.execute(sql_select)
        id_get = self.my_cursor.fetchone()
        if id_get is None:
            print("登录账号错误\n")
            return -1, False
        else:
            password_get = id_get[2]
            print("pw: ", password_get)
            if password == password_get:
                return 1, True
            else:
                print("密码错误\n")
                return -2, False
        pass

    # 注册
    # 数据需要按照指定格式传入
    # 没有数据格式的判断
    def user_log_up(self, phoneNum, name, password):
        sql_select = "select * from [user] where id='%s'" % phoneNum
        self.my_cursor.execute(sql_select)
        result = self.my_cursor.fetchone()
        if result is None:
            sql_insert = "INSERT INTO [user] VALUES ('%s','%s','%s')" % (phoneNum, name, password)
            self.my_cursor.execute(sql_insert)
            print("插入语句", sql_insert)
            self.my_connect.commit()
            return 1
        else:
            print("账号重复\n")
            return -1
        pass

    # 修改用户信息
    # 数据需要按照指定格式传入
    # 没有数据格式的判断
    def update_user(self, userID, userName, password):
        sql_select = "select * from [user] where id='%s'" % userID
        self.my_cursor.execute(sql_select)
        select_result = self.my_cursor.fetchone()
        if select_result is None:
            print("找不到用户\n")
            return -1, False
        else:
            sql_update = "update [user] set name='%s', password='%s' where id='%s'" % (userName, password, userID)
            self.my_cursor.execute(sql_update)
            self.my_connect.commit()
            print("修改成功\n")
            return 1, True
        pass

    # 添加一条项目历史数据
    # c = char
    # vc = varchar
    # bit = binary
    def add_project(self, id_c4, name_vc20, function_int2, deadline_date, user_id_c11, image1='', image2='', image3=''):
        sql_select_user = "select * from [user] where id='%s'" % user_id_c11
        self.my_cursor.execute(sql_select_user)
        user_select_result = self.my_cursor.fetchone()
        if user_select_result is None:
            print(sql_select_user)
            print("找不到用户\n")
            return -1, False
        else:
            sql_select_project = "select * from [project] where id='%s' and user_id='%s'" % (id_c4, user_id_c11)
            self.my_cursor.execute(sql_select_project)
            project_select_result = self.my_cursor.fetchone()
            if project_select_result is None:
                sql_insert_project = "insert into [project] values ('%s', '%s', '%d', '%s', '%s', '%s', '%s', '%s')" \
                                     % (id_c4,
                                        name_vc20,
                                        function_int2,
                                        deadline_date,
                                        user_id_c11,
                                        image1,
                                        image2,
                                        image3)
                self.my_cursor.execute(sql_insert_project)
                self.my_connect.commit()
                print("添加项目成功\n")
                return 1, True
                pass
            else:
                print("项目主键重复\n")
                return -2, False
        pass

    # 根据用户id获取所有的项目表
    def request_project(self, user_id_c11):
        sql_select_user = "select * from [project] where user_id='%s'" % user_id_c11
        self.my_cursor.execute(sql_select_user)
        user_select_result = self.my_cursor.fetchall()
        if user_select_result is None:
            print("找不到用户\n")
            return -1, None
        else:
            paths = []
            for item in user_select_result:
                paths.append([item[0], item[2], item[3], item[4], item[5], item[6], item[7]])
            print(paths)
            return 1, paths
        pass

    # 删除项目
    def delete_project(self, id_c4, user_id_c11):
        sql_select_project = "select * from [project] where id='%s' and user_id='%s'" % (id_c4, user_id_c11)
        self.my_cursor.execute(sql_select_project)
        project_select_result = self.my_cursor.fetchone()
        if project_select_result is None:
            print("找不到该项目")
        else:
            sql_delete_project = "delete from [project] where id='%s' and user_id='%s'" % (id_c4, user_id_c11)
            self.my_cursor.execute(sql_delete_project)
            self.my_connect.commit()
            print("删除成功")
            print(project_select_result)
            return True
        pass

    def __del__(self):
        self.my_cursor.close()
        self.my_connect.close()


if __name__ == '__main__':
    db = db_operator()
    db.user_log_up('33333333334', 'lxx', '12233445')
    # db.update_user('33333333334', 'lxx_xx', '1221123123')
    db.add_project("0002", '00001', 1, '2022-05-20', '33333333334')
    db.request_project("33333333334")
    # db.delete_project('0003', '33333333334')
    # print(db.log_in('33333333333', '1223344'))
    # cursor.execute("select * from [user] ")
    # cnt = cursor.fetchone()[0]
    # print(cnt)
    #
    # cursor.execute("update [user] set password='666666' where id='19218952238' ")
    # conn.commit()
