import pyvjoy
import cv2
from PIL import ImageGrab
import numpy as np
import threading
import paddle.fluid as fluid

rect = (678, 671, 1078, 871)
j = pyvjoy.VJoyDevice(1)
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname="./model_infer/", executor=exe)

def control(ang):
    if ang > 60:
        ang = 60
    if ang < -60:
        ang = -60
    global j
    x = ang / 180 + 0.5
    j.data.wAxisX = int(x * 32767)
    j.data.wAxisY = int(0 * 32767)
    j.data.wAxisZ = 0
    j.update()

def img_process(img):
    # 统一图片大小
    img = cv2.resize(img,(120,120))
    # 把图片转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 归一化
    img = img[(2, 1, 0), :, :] / 255.0
    # 增加维度
    img = np.expand_dims(img, axis=0)
    return img

def get_img():
    global rect
    while True:
        im = ImageGrab.grab(rect)  # 获得当前屏幕
        imm = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)  # 转为opencv的BGR格式
        img = img_process(imm)
        result = exe.run(program=infer_program, feed={feeded_var_names[0]: img}, fetch_list=target_var)
        angle = result[0][0][0]
        control(angle)
        print("angle:", angle)
        cv2.imshow('image', imm)  # 显示
        cv2.waitKey(3)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Thread2 = threading.Thread(target=get_img)
    Thread2.start()
