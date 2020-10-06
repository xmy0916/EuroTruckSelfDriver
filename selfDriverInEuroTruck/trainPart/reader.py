'''
get image from     train_list_path  test_list_path
'''

import os
import random
from multiprocessing import cpu_count
import numpy as np
import paddle
from PIL import Image
import cv2 as cv

# 训练图片的预处理
def train_mapper(sample):
    img_path, label, crop_size, resize_size = sample
    # try:
    img = Image.open(img_path)
    # 统一图片大小
    img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
    # 把图片转换成numpy值
    img = np.array(img).astype(np.float32)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img, label
    # except:
    #     print("%s 该图片错误，请删除该图片并重新创建图像数据列表" % img_path)


# 获取训练的reader
def train_reader(train_list_path, crop_size, resize_size):
    father_path = os.path.dirname(train_list_path)

    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            # 打乱图像列表
            np.random.shuffle(lines)
            # 开始获取每张图像和标签
            for line in lines:
                img = line.split(' ')[0]
                label = line.split(' ')[1:3]
                label[1] = label[1].replace("\n","")
                img = os.path.join(father_path, img)
                yield img, label, crop_size, resize_size

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 102400)

# 测试图片的预处理
def test_mapper(sample):
    img, label, crop_size = sample
    img = Image.open(img)
    # 统一图像大小
    img = img.resize((crop_size, crop_size), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img, label

# 测试的图片reader
def test_reader(test_list_path, crop_size):
    father_path = os.path.dirname(test_list_path)

    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img = line.split(' ')[0]
                label = line.split(' ')[1:2]
                img = os.path.join(father_path, img)
                yield img, label, crop_size

    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 1024)


