from flask import Flask, render_template, jsonify, url_for, make_response, request
import time
from change_detection import mainFunc_cd
from ground_classification import mainFunc_gc
from target_detection import mainFunc_td
from target_extract import mainFunc_te
import DB_operate
import json
from gevent import pywsgi
import socket

# 获取本机ip
hostname = socket.gethostname()
# ip = socket.gethostbyname(hostname)
ip = '127.0.0.1'
soc = '5000'
ipsocket = 'http://' + ip + ':' + soc

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# 是否登录标志
db = DB_operate.db_operator()


# 主界面
@app.route('/<usrid>')
def index(usrid):
    return render_template('index.html', logined=usrid, ipsocket=ipsocket)


# 登录
@app.route('/login')
def login():
    return render_template('login.html', ipsocket=ipsocket)


# 登录中
@app.route('/logining')
def logining():
    id = request.args.get('id')
    pwd = request.args.get('pwd')
    code, flag = db.user_log_in(id, pwd)
    if code == -1:
        return jsonify({'msg': '-1', 'url': ipsocket})
    elif code == 1:
        return jsonify({'msg': '1', 'url': ipsocket})
    else:
        return jsonify({'msg': '-2', 'url': ipsocket})


# 注册
@app.route('/signup')
def signup():
    return render_template('signup.html', ipsocket=ipsocket)


# 注册
@app.route('/signuping')
def signuping():
    id = request.args.get('id')
    name = request.args.get('name')
    pwd = request.args.get('pwd')
    flag = db.user_log_up(id, name, pwd)
    return jsonify({'msg': flag, 'url': ipsocket})


# 个人界面
@app.route('/private/<usrid>')
def private(usrid):
    return render_template('introduction.html', usrid=usrid, ipsocket=ipsocket)


# 获取个人项目信息
#item: id functiontype deadline owner img1 img2 igm3
@app.route('/getPrivateData/<usrid>')
def getPrivateData(usrid):
    flag, path = db.request_project(usrid)
    if flag == -1:
        return jsonify({'msg': '-1'})
    else:
        msg_map = {}
        index = 2
        msg_map[0] = len(path)
        msg_map[1] = usrid
        for item in path:
            msg_map[index] = item[0]
            msg_map[index + 1] = item[1]
            msg_map[index + 2] = item[2]
            msg_map[index + 3] = url_for('static', filename=item[3])
            msg_map[index + 4] = url_for('static', filename=item[4])
            msg_map[index + 5] = url_for('static', filename=item[5])
            msg_map[index + 6] = url_for('static', filename=item[6])
            index = index + 7
    return jsonify(msg_map)


#删除项目
@app.route('/deleteProject/<usrid>')
def deleteProject(usrid):
    projectId = request.args.get('projectID')
    db.delete_project(projectId, usrid)
    return jsonify({'url': ipsocket+'/private/'+usrid})


# {'1':'变化检测',
#  '2':'地物分类',
#  '3':'目标检测',
#  '4':'目标提取'};

# 变化检测
@app.route('/function_cd/<usrid>')
def function_cd(usrid):
    return render_template('function_cd.html', usrid=usrid, ipsocket=ipsocket)


# 变化检测
@app.route('/functionDealing_cd/<usrid>', methods=['GET', 'POST'])
def functionDealing_cd(usrid):
    img = request.files.get('uploadImg1')  # 从post请求中获取图片数据
    imgPathHead1 = 'imgBase/' + str(int(time.time())) + 'img1_cd.' + img.filename.split('.')[-1]
    img_path1 = 'static/' + imgPathHead1  # 拼接图片完整保存路径,时间戳命名文件防止重复
    img.save(img_path1)  # 保存图片

    img = request.files.get('uploadImg2')  # 从post请求中获取图片数据
    imgPathHead2 = 'imgBase/' + str(int(time.time())) + 'img2_cd.' + img.filename.split('.')[-1]
    img_path2 = 'static/' + imgPathHead2  # 拼接图片完整保存路径,时间戳命名文件防止重复
    img.save(img_path2)  # 保存图片

    result = mainFunc_cd(img_path1, img_path2)

    db.add_project(str(int(time.time())), 'project', 1, '2025-11-11', usrid, imgPathHead1, imgPathHead2, result)
    return jsonify({'url': url_for('static', filename=result)})


# 地物分类
@app.route('/function_gc/<usrid>')
def function_gc(usrid):
    return render_template('function_gc.html', usrid=usrid, ipsocket=ipsocket)


# 地物分类
@app.route('/functionDealing_gc/<usrid>', methods=['GET', 'POST'])
def functionDealing_gc(usrid):
    img = request.files.get('uploadImg1')  # 从post请求中获取图片数据
    imgPathHead1 = 'imgBase/' + str(int(time.time())) + 'img_gc.' + img.filename.split('.')[-1]
    img_path1 = 'static/' + imgPathHead1  # 拼接图片完整保存路径,时间戳命名文件防止重复
    img.save(img_path1)  # 保存图片

    result = mainFunc_gc(img_path1)
    db.add_project(str(int(time.time())), 'project', 2, '2025-11-11', usrid, imgPathHead1, '', result)
    return jsonify({'url': url_for('static', filename=result)})

# 目标检测
@app.route('/function_td/<usrid>')
def function_td(usrid):
    return render_template('function_td.html', usrid=usrid, ipsocket=ipsocket)


# 目标检测
@app.route('/functionDealing_td/<usrid>', methods=['GET', 'POST'])
def functionDealing_td(usrid):
    img = request.files.get('uploadImg1')  # 从post请求中获取图片数据
    imgPathHead1 = 'imgBase/' + str(int(time.time())) + 'img_td.' + img.filename.split('.')[-1]
    img_path1 = 'static/' + imgPathHead1  # 拼接图片完整保存路径,时间戳命名文件防止重复
    img.save(img_path1)  # 保存图片

    result = mainFunc_td(img_path1)
    db.add_project(str(int(time.time())), 'project', 3, '2025-11-11', usrid, imgPathHead1, '', result)
    return jsonify({'url': url_for('static', filename=result)})


# 目标提取
@app.route('/function_te/<usrid>')
def function_te(usrid):
    return render_template('function_te.html', usrid=usrid, ipsocket=ipsocket)


# 目标提取
@app.route('/functionDealing_te/<usrid>', methods=['GET', 'POST'])
def functionDealing_te(usrid):
    img = request.files.get('uploadImg1')  # 从post请求中获取图片数据
    imgPathHead1 = 'imgBase/' + str(int(time.time())) + 'img_te.' + img.filename.split('.')[-1]
    img_path1 = 'static/' + imgPathHead1  # 拼接图片完整保存路径,时间戳命名文件防止重复
    img.save(img_path1)  # 保存图片

    result = mainFunc_te(img_path1)
    db.add_project(str(int(time.time())), 'project', 4, '2025-11-11', usrid, imgPathHead1, '', result)
    return jsonify({'url': url_for('static', filename=result)})


@app.route('/test3')
def test3():
    resp = make_response(jsonify({'url': 'http://127.0.0.1:5000' + url_for('static', filename='image/10.png')}))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/send_message', methods=['GET', 'POST'])
def send_message():
    return jsonify({'url': url_for('static', filename='image/10.png')})


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()
