# -*- coding: utf-8 -*-
from utils.util import AttrDict, merge_cfg_from_args, get_arguments
import os

args = get_arguments()
args.example = "Road"
cfg = AttrDict()

cfg.model = "2"
cfg.mode = 1
# 待预测图像所在路径
cfg.data_dir = os.path.join(args.example , "data", "test_images")
# 待预测图像名称列表
cfg.data_list_file = os.path.join(args.example , "data", "test.txt")
# 模型加载路径
cfg.model_path = os.path.join(args.example , "model/" + cfg.model)
# 预测结果保存路径
cfg.vis_dir = os.path.join(args.example , "result")

# 预测类别数
cfg.class_num = 19
# 均值, 图像预处理减去的均值
cfg.MEAN = 127.5, 127.5, 127.5
# 标准差，图像预处理除以标准差
cfg.STD =  127.5, 127.5, 127.5
# 待预测图像输入尺寸

cfg.input_size = 400, 200


merge_cfg_from_args(args, cfg)
