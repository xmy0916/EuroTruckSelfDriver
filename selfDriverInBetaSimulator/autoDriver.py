# -*- coding:utf-8 -*-



import paddle.fluid as fluid

# 通信接口
import socketio
import eventlet
import eventlet.wsgi
# 网络框架
from flask import Flask,render_template

from PIL import Image
import numpy as np
from io import BytesIO
import base64
import cv2


sio = socketio.Server() # 启动服务器
app = Flask(__name__) # 创建网络框架app
size = (120,120)


place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname="./model_infer/", executor=exe)

kp = 0.5
ki = 0.01
kd = 0.3
last = 0.0
sum = 0.0

def PID_speed(now,target):
    global last,sum
    error = target - now
    out = kp * error + kd * (error - last) + ki * sum
    last = error
    sum += error
    if sum > 100 :
        sum = 100
    if sum < -100:
        sum = -100
    return out

@sio.on('telemetry')
def telemetry(sid,data):
    try:
        # steering_angle = float(data["steering_angle"])
        # throttle = float(data["throttle"])
        now_speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image_array = np.asarray(image)  # from PIL image to numpy array
        image_array = cv2.cvtColor(image_array,cv2.COLOR_RGB2BGR)
        img = img_process(image_array)
        result = exe.run(program=infer_program, feed={feeded_var_names[0]: img}, fetch_list=target_var)
        angle = result[0][0][0]
        print("angle:",angle)
        cv2.imshow("1",image_array)
        cv2.waitKey(3)
        steer = (90 - angle) * 3.1415926 / 180
        speed = PID_speed(now_speed,10)
        send_control(steer,speed)
    except:
        print("手动！")
        send_control(0, 0)




@sio.on('connect')
def connect(sid,environ):
    print('connect!',sid)
    send_control(0,0)


def send_control(steer,speed):
    sio.emit("steer",data={
        'steering_angle':steer.__str__(),
        'throttle':speed.__str__()
    },skip_sid=True)


def img_process(img):
    # 统一图片大小
    img = cv2.resize(img,(120,120))
    # 把图片转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)
    return img


if __name__ == '__main__':
    app = socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(('',4567)),app)