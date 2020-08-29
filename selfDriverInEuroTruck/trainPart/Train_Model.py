'''
get data from ../data/test.list  ../data/train.list   ../data/image/{}.jpg

put model  to ../model/model_infer
'''

import os
import shutil
#import mobilenet_v1
import cnn_model
import paddle as paddle
import reader
import paddle.fluid as fluid
import numpy as np
from sys import argv
import getopt
import matplotlib.pyplot as plt



path = os.path.split(os.path.realpath(__file__))[0]+"/.."
opts,args = getopt.getopt(argv[1:],'-hH',['test_list=','train_list=','save_path='])

test_list = "test_data.txt"
train_list = "train_data.txt"
save_path = "../model_infer"



for opt_name,opt_value in opts:
    if opt_name in ('-h','-H'):
        print("python3 Train_Model.py  --test_list=%s   --train_list=%s  --save_path=%s  "%(test_list , train_list , save_path))
        exit()

    if opt_name in ('--test_list'):
        test_list  = opt_value

    if opt_name in ('--train_list'):
        train_list = opt_value
        
    if opt_name in ('--save_path'):
        save_path = opt_value

   
test_list  = test_list
train_list  = train_list
save_path  = save_path

iter=0
iters=[]
train_costs=[]

def draw_train_process(iters,train_costs):
    title="training cost"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.ylim(0, 100)
    plt.plot(iters, train_costs,color='red',label='training cost') 
    plt.grid()
    plt.savefig("filename.png")
    plt.show()
    

crop_size = 120
resize_size = 120


image = fluid.layers.data(name='image', shape=[3, crop_size, crop_size], dtype='float32')
label = fluid.layers.data(name='label', shape=[2], dtype='float32')

model = cnn_model.cnn_model(image)

cost = fluid.layers.square_error_cost(input=model, label=label)
avg_cost = fluid.layers.mean(cost)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)

opts = optimizer.minimize(avg_cost)


# 获取自定义数据
train_reader = paddle.batch(reader=reader.train_reader(train_list, crop_size, resize_size), batch_size=32)
test_reader = paddle.batch(reader=reader.test_reader(test_list, crop_size), batch_size=32)

# 定义执行器
place = fluid.CUDAPlace(0)  # place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])


# 训练
all_test_cost = []
for pass_id in range(100):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(program=fluid.default_main_program(),
                            feed=feeder.feed(data),
                            fetch_list=[avg_cost])
        train_costs.append(train_cost[0][0])
        iter=iter+32
        iters.append(iter)
        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f' %
                  (pass_id, batch_id, train_cost[0]))

    # 进行测试
    test_costs = []

    for batch_id, data in enumerate(test_reader()):
        test_cost = exe.run(program=test_program,
                            feed=feeder.feed(data),
                            fetch_list=[avg_cost])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    all_test_cost.append(test_cost)


    #test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f' % (pass_id, test_cost))
    save_path = '../model_infer'
    # 保存预测模型

    if min(all_test_cost) >= test_cost:
        fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], main_program=test_program, target_vars=[model], executor=exe)
        print('finally test_cost: {}'.format(test_cost))

draw_train_process(iters,train_costs)