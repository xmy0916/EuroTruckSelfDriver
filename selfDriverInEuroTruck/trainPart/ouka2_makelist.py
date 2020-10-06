import glob
import os


# 数据路径
data_path = "./dataset/"
trainList = './train_data.txt'
testList = './test_data.txt'
# 多少比例用作训练集
ratio = 0.8


# 只读csv文件
with open(data_path + "log_remap.txt",'r') as logFile:
    _list = logFile.readlines()

    # 判断图片数是否匹配
    # ls_imgs = glob.glob(data_path + 'IMG/*.jpg')
    # print(len(ls_imgs))
    # print(len(_list))
    # assert len(ls_imgs) == len(_list),'number of images does not match'


    if (os.path.exists(trainList)):
        os.remove(trainList)
    if (os.path.exists(testList)):
        os.remove(testList)

    with open(trainList, 'a') as f_train:
        with open(testList, 'a') as f_test:
            for index,_list in enumerate(_list):
                ang = int(_list.split(" ")[1])
                brake = float(_list.split(" ")[2]) * 10
                img = _list.split(" ")[0]
                if index % (int(10*ratio) + 1) == 0:
                    f_test.write(img + " " + str(ang) + " " + str(round(brake,2)) + "\n")
                else:
                    f_train.write(img + " " + str(ang) + " " + str(round(brake,2)) + "\n")

    print("生成完毕，路径：" + testList + " | " +trainList)