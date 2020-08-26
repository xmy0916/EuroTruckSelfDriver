# 项目使用说明
## 简介
本项目基于paddlepaddle深度学习框架，使用PIL库录屏的方式获取欧卡2游戏的视觉数据，搭建卷积神经网络拟合方向盘的角度，输入是视觉数据，输出是方向盘角度。通过vjoy虚拟摇杆来控制游戏中的卡车，实现了在欧卡2游戏中高速上无人驾驶。

## 效果演示
视屏地址：[传送门](https://www.bilibili.com/video/BV1gC4y1t7YY)

## 项目使用
- selfDriverInBetaSimulator文件夹：在MIT仿真软件UDACITY上实现的无人驾驶代码。
- selfDriverInEuroTruck文件夹：在欧卡2游戏里实现无人驾驶的代码。

## 欧卡2无人驾驶快速使用教程
### step1
安装software文件夹下的vJoySetup.exe，双击运行即可，一直默认安装。
### step2
修改欧卡2的控制器，在选项->控制器->设置键盘加vjoy。
### step3
运行autoDriver.py脚本开始自动驾驶。
！！！重要👇
>脚本运行后会有opencv的窗口出现，需要调整你的游戏界面使得窗口内的图像是你的车窗视角

## 欧卡2无人驾驶完整训练加数据集准备教程
### step1
安装software文件夹下的vJoySetup.exe，双击运行即可，一直默认安装。
### step2
修改欧卡2的控制器，在选项->控制器->设置键盘加vjoy。
### step3
运行recordData.py脚本录制训练数据：
！！！重要👇
>按键‘O’开始录制
按键‘B’退出录制并写入log
未开始录制时按键a d控制方向盘
开启录制后按键u i控制方向盘

### step4
运行trainPart/ouka2_makelist.py文件生成train_data.txt和test_data.txt。
### step5
运行trainPart/Train_Model.py开始训练，训练结束会自动保存在model_infer文件夹中。
### step6
运行autoDriver.py脚本开始自动驾驶。
！！！重要👇
>脚本运行后会有opencv的窗口出现，需要调整你的游戏界面使得窗口内的图像是你的车窗视角