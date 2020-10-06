import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from PIL import Image
import random


def draw_from_dict(dicdata,RANGE, heng=0):
    #dicdata：字典的数据。
    #RANGE：截取显示的字典的长度。
    #heng=0，代表条状图的柱子是竖直向上的。heng=1，代表柱子是横向的。考虑到文字是从左到右的，让柱子横向排列更容易观察坐标轴。
    list_key = list(dicdata.keys())
    list_key.sort()
    #by_value = sorted(dicdata.items(),key = lambda item:item[1],reverse=True)
    x = []
    y = []
    #print(by_value)

    for d in list_key:
        #print(dicdata[d])
        x.append(str(d))
        y.append(dicdata[d])

    plt.title("")
    if heng == 0:
        plt.tick_params(labelsize=6)
        plt.bar(x[0:RANGE], y[0:RANGE],0.25)
        plt.show()
        return
    elif heng == 1:
        plt.tick_params(labelsize=6)
        plt.barh(x[0:RANGE], y[0:RANGE],0.25)
        plt.show()
        return
    else:
        return "heng的值仅为0或1！"


dict = {}
leng = 0
with open("./dataset/log.txt",'r') as v:
    reader = v.readlines()
    for line in reader:
        angle = line.split(" ")[1]
        speed = line.split(" ")[2]
        path = line.split(" ")[0]
        if int(angle) not in dict.keys():
            dict[int(angle)] = 1
        else:
            dict[int(angle)] += 1

print(dict)
draw_from_dict(dict,len(dict))

if os.path.exists("./dataset/log_remap.txt"):
    os.remove("./dataset/log_remap.txt")

with open("./dataset/log_remap.txt",'w') as t:
    total = dict[0]
    for line in reader:
        path = line.split(" ")[0]
        angle = int(line.split(" ")[1])
        speed = line.split(" ")[2]
        if angle == 0:
            x = random.uniform(0, 1)
            if x < 250/total:
                t.write(line)
            else:
                dict[0] -= 1
        else:
            t.write(line)
            if dict[-angle] < 250:
                img = Image.open(path)
                filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                new_name = "flp_" + path.split('/')[-1]
                filp_img.save("./dataset/IMG/" + new_name)
                t.write("./dataset/IMG/" + new_name + " " + str(-angle) + " " + speed)
                dict[-angle] += 1
print(dict)
draw_from_dict(dict,len(dict))
