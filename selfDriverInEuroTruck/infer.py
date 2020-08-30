# -*- coding: utf-8 -*-
from utils.util import get_arguments
from utils.palette import get_palette
from PIL import Image as PILImage
import importlib
from PIL import ImageGrab
import threading
import numpy as np
import cv2
import time
import os
import shutil
import pyvjoy

args = get_arguments()
args.example = "Road"
config = importlib.import_module(args.example + '.config')
cfg = getattr(config, 'cfg')
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'

import paddle.fluid as fluid

class myRecord:
    def __init__(self,width,height,imm):
        self.recordFlag = 0
        self.data_dir = cfg.data_dir
        self.data_list_file = cfg.data_list_file
        self.data_list = self.get_data_list()
        self.data_num = len(self.data_list)
        if not os.path.exists("config.txt"):
            os.system(r"touch {}".format("config.txt"))
        f = open('config.txt',"r")
        line = f.readline()
        if len(line) != 0:
            self.left =  int(line.split(" ")[0])
            self.top = int(line.split(" ")[1])
            self.right = int(line.split(" ")[2])
            self.bottom = int(line.split(" ")[3])
        else:
            self.left, self.right, self.top, self.bottom = 0,width,0,height
        f.close()
        self.imm = imm
        self.width = width
        self.height = height
        self.j = pyvjoy.VJoyDevice(1)
        self.control(0)

        self.recordThread1 = threading.Thread(target=self.record)
        self.recordThread2 = threading.Thread(target=self.infer)
        self.recordThread3 = threading.Thread(target=self.infer_vedio)
        if cfg.mode == 1:
            self.recordThread1.start()
            self.recordThread2.start()
        elif cfg.mode == 2:
            self.recordThread3.start()
        self.setDir("result/")


    def record(self):
        cv2.namedWindow('image')

        # 是否显示修改窗口大小的框
        changeWindowSize = False
        if changeWindowSize:
            cv2.namedWindow('tool')
            cv2.createTrackbar('left', 'tool', self.left, self.width, self.left_callback)
            cv2.createTrackbar('top', 'tool', self.top, self.height, self.top_callback)
            cv2.createTrackbar('right', 'tool', self.right , self.width, self.right_callback)
            cv2.createTrackbar('bottom', 'tool', self.bottom, self.height, self.bottom_callback)

        fps = 16  # 视频每秒24帧
        size = (self.right-self.left,self.bottom-self.top)  # 需要转为视频的图片的尺寸
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码格式
        video = cv2.VideoWriter('./vedio/record.avi', fourcc, fps, size)
        while True:
            im = ImageGrab.grab((self.left,self.top,self.right,self.bottom))  # 获得当前屏幕
            self.imm = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)  # 转为opencv的BGR格式
            cv2.imshow('image', self.imm)  # 显示

            if self.recordFlag == 1:
                video.write(self.imm)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # q键推出
                break
            elif key == ord('s'):  # q键推出
                print("开始录制")
                self.recordFlag = 1

        video.release()
        cv2.destroyAllWindows()



    def infer(self):
        if not os.path.exists(cfg.vis_dir):
            os.makedirs(cfg.vis_dir)
        palette = get_palette(cfg.class_num)

        # 使用GPU
        cfg.use_gpu = True
        place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)

        # 加载预测模型
        test_prog, feed_name, fetch_list = fluid.io.load_inference_model(
            dirname=cfg.model_path, executor=exe, params_filename='__params__')

        while True:
            # 数据获取
            ori_img = self.imm
            image = self.preprocess(ori_img)
            im_shape = ori_img.shape[:2]

            #模型预测
            last_time = time.time()
            result = exe.run(program=test_prog, feed={feed_name[0]: image}, fetch_list=fetch_list, return_numpy=True)
            print("推理延迟：" + str(round(time.time() - last_time, 2)))
            parsing = np.argmax(result[0][0], axis=0)
            parsing = cv2.resize(parsing.astype(np.uint8), im_shape[::-1])

            # 预测结果
            output_im = PILImage.fromarray(np.asarray(parsing, dtype=np.uint8))
            output_im.putpalette(palette)
            arr = np.asarray(output_im)
            # img2infer = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) # 用来推理的图像
            # 推理的代码

            arr = np.where(arr == 13, 125, arr)
            arr = np.where(arr == 0, 255, arr)
            arr = np.where(arr < 30, 30, arr)
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # 用来显示的图片
            cv2.imshow("cvPicture", img)
            cv2.waitKey(3)



    def infer_vedio(self):
        if not os.path.exists(cfg.vis_dir):
            os.makedirs(cfg.vis_dir)
        palette = get_palette(cfg.class_num)

        cfg.use_gpu = True
        place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)

        # 加载预测模型
        test_prog, feed_name, fetch_list = fluid.io.load_inference_model(
            dirname=cfg.model_path, executor=exe, params_filename='__params__')

        # 加载预测数据集
        last_time = time.time()

        cap = cv2.VideoCapture("./vedio/record.avi")  # 打开视频
        size = (self.right-self.left,self.bottom-self.top)  # 需要转为视频的图片的尺寸
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码格式
        video = cv2.VideoWriter("./vedio/" + cfg.model + '_result.avi', fourcc, 16, size)
        while cap.isOpened():
            # 数据获取
            ret, fram = cap.read()
            ori_img = fram
            if ori_img is None:
                return
            image = self.preprocess(ori_img)
            im_shape = ori_img.shape[:2]

            # HumanSeg,RoadLine模型单尺度预测
            start_time = time.time()
            result = exe.run(program=test_prog, feed={feed_name[0]: image}, fetch_list=fetch_list, return_numpy=True)
            print("推理延迟：" + str(round(time.time() - start_time, 2)))
            parsing = np.argmax(result[0][0], axis=0)
            parsing = cv2.resize(parsing.astype(np.uint8), im_shape[::-1])

            # 预测结果保存
            output_im = PILImage.fromarray(np.asarray(parsing, dtype=np.uint8))
            output_im.putpalette(palette)
            img = cv2.cvtColor(np.asarray(output_im) * 255, cv2.COLOR_RGB2BGR)  # PIL转cv

            video.write(img)
            cv2.imshow("2", img)
            cv2.waitKey(1)


    def setDir(self,filepath):
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            shutil.rmtree(filepath)
            os.mkdir(filepath)


    def get_data_list(self):
        # 获取预测图像路径列表
        data_list = []
        data_file_handler = open(self.data_list_file, 'r',encoding='utf-8')
        for line in data_file_handler:
            img_name = line.strip()
            name_prefix = img_name.split('.')[0]
            if len(img_name.split('.')) == 1:
                img_name = img_name + '.jpg'
            img_path = os.path.join(self.data_dir, img_name)
            data_list.append(img_path)
        return data_list

    def preprocess(self, img):
        # 图像预处理
        img = cv2.resize(img, cfg.input_size).astype(np.float32)
        if cfg.model in "mobilenet_hsv":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img -= np.array(cfg.MEAN)
        img /= np.array(cfg.STD)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def control(self,angle):
        if angle > 60:
            angle = 60
        if angle < -60:
            angle = -60

        x = angle / 180 + 0.5
        self.j.data.wAxisX = int(x * 32767)
        self.j.data.wAxisY = int(0.4 * 32767)
        self.j.data.wAxisZ = 0
        self.j.update()

    def get_data(self):
        # 获取图像信息
        img = self.imm
        if img is None:
            return img, img, "", None

        img_shape = img.shape[:2]
        img_process = self.preprocess(img)

        return img, img_process, "", img_shape


    def left_callback(self,x):
        self.left = x
        if self.left >= self.right:
            self.left = self.right - 5
        f = open('config.txt',"w")
        f.write("{} {} {} {}".format(self.left,self.top,self.right,self.bottom))
        f.close()

    def right_callback(self,x):
        self.right = self.width - x
        if self.left >= self.right:
            self.right = self.left + 5
        f = open('config.txt',"w")
        f.write("{} {} {} {}".format(self.left,self.top,self.right,self.bottom))
        f.close()

    def top_callback(self, x):
        self.top = x
        if self.top >= self.bottom:
            self.top = self.bottom - 5
        f = open('config.txt',"w")
        f.write("{} {} {} {}".format(self.left,self.top,self.right,self.bottom))
        f.close()

    def bottom_callback(self, x):
        self.bottom = self.height - x
        if self.top >= self.bottom:
            self.bottom = self.top + 5
        f = open('config.txt',"w")
        f.write("{} {} {} {}".format(self.left,self.top,self.right,self.bottom))
        f.close()


if __name__ == "__main__":
    window = ImageGrab.grab()  # 获得当前屏幕,存窗口大小
    imm = cv2.cvtColor(np.array(window), cv2.COLOR_RGB2BGR)  # 转为opencv的BGR格式
    width,height = window.size
    r = myRecord(width,height,imm)
