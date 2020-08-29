# -*- coding:utf-8 -*-
import numpy as np
import cv2



import paddle.fluid as fluid
from PIL import Image


def img_process(path):
    img = Image.open(path)
    # 统一图片大小
    img = img.resize((120, 120), Image.ANTIALIAS)
    # 把图片转换成numpy值
    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)
    return img



if __name__ == "__main__":
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname="../model_infer/", executor=exe)

    with open("test_data.txt",'r') as t:
        all_lines = t.readlines()
        for line in all_lines:
            imgPath = line.split(" ")[0]
            ang = float(line.split(" ")[1])

            img = img_process(imgPath)
            result = exe.run(program=infer_program,feed={feeded_var_names[0]: img},fetch_list=target_var)
            print(result[0][0])
            #angle = result[0][0][0]
            #print("name:%s pre:%.5f true:%.5f"%(imgPath,angle,ang))
