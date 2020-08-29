import win32api
import time
import pyvjoy
import cv2
from PIL import ImageGrab
import numpy as np
import threading
import os

angle = 0
brake = 0.0
rect = (678, 671, 1078, 871) # 捕获窗口大小
j = pyvjoy.VJoyDevice(1)
recordFlag = False
stopFlag = False

def control(ang,brake):
    if ang > 60:
        ang = 60
    if ang < -60:
        ang = -60
    global j
    x = ang / 180 + 0.5
    j.data.wAxisX = int(x * 32767)
    j.data.wAxisY = int(brake * 32767) # 0改成0.4可以低速运行
    j.data.wAxisZ = 0
    j.update()

def get_angle():
    key1 = ord('U')
    key2 = ord('I')
    key5 = ord('G')
    key3 = ord('O')
    key4 = ord('B')
    global recordFlag,stopFlag

    global angle,brake
    def isKeyPressed(key):
        return (win32api.GetKeyState(key) & (1 << 7)) != 0

    print("INFO:\n按键‘O’开始录制\n按键‘B’退出录制并写入log\n未开始录制时按键a d控制方向盘\n开启录制后按键u i控制方向盘\nG刹车")
    while not isKeyPressed(key3):
        time.sleep(0.01)

    recordFlag = True
    wasKeyPressedTheLastTimeWeChecked = False
    wasKeyPressedTheLastTimeWeChecked2 = False
    wasKeyPressedTheLastTimeWeChecked5 = False
    left = False
    right = False
    stop = False
    while not stopFlag:
        if isKeyPressed(key4):
            stopFlag = True
        keyIsPressed = isKeyPressed(key1)
        if keyIsPressed and not wasKeyPressedTheLastTimeWeChecked:
            left = True
        if not keyIsPressed and wasKeyPressedTheLastTimeWeChecked:
            left = False
            angle = 0
        wasKeyPressedTheLastTimeWeChecked = keyIsPressed

        if left is True:
            angle -= 2
            if angle < -60:
                angle = -60

        time.sleep(0.01)

        keyIsPressed2 = isKeyPressed(key2)
        if keyIsPressed2 and not wasKeyPressedTheLastTimeWeChecked2:
            right = True
        if not keyIsPressed2 and wasKeyPressedTheLastTimeWeChecked2:
            right = False
            angle = 0
        wasKeyPressedTheLastTimeWeChecked2 = keyIsPressed2

        if right is True:
            angle += 2
            if angle > 60:
                angle = 60
        time.sleep(0.01)

        keyIsPressed5 = isKeyPressed(key5)
        if keyIsPressed5 and not wasKeyPressedTheLastTimeWeChecked5:
            stop = True
        if not keyIsPressed5 and wasKeyPressedTheLastTimeWeChecked5:
            stop = False
            brake = 0
        wasKeyPressedTheLastTimeWeChecked5 = keyIsPressed5

        if stop is True:
            brake += 0.05
            if brake > 1:
                brake = 1
        time.sleep(0.01)

        control(angle,brake)
        #print("angle = ",angle)
        print("angle = %d  brake = %f" %(angle,brake))


def get_img():
    global recordFlag,stopFlag,rect
    imgpath = "./result/IMG/"
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)
    list2write = []

    while not stopFlag:
        im = ImageGrab.grab(rect)  # 获得当前屏幕
        imm = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)  # 转为opencv的BGR格式
        if recordFlag == True:
            imgName = imgpath + str(time.time()) + '.jpg'
            cv2.imwrite(imgName, imm)
            list2write.append(imgName + " " + str(angle) + " " + str(round(brake, 2)) + '\n')
        cv2.imshow('image', imm)  # 显示
        cv2.waitKey(3)

    cv2.destroyAllWindows()
    fwrite = open('./result/log.txt', 'w')
    for item in list2write:
        fwrite.write(item)
    fwrite.close()


if __name__ == '__main__':
    Thread1 = threading.Thread(target=get_angle)
    Thread2 = threading.Thread(target=get_img)
    Thread1.start()
    Thread2.start()
