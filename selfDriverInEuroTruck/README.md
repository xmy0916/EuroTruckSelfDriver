# paddleseg城市景物分割
本例子使用DeepLabv3网络，为了追求推理速度backbone选择MobilenetV2，项目的实现参考百度AIStudio的公开项目：PaddleSeg_DeepLabv3+。

# 本项目使用说明
因为github提交文件大小受限制，模型文件我上传了百度网盘，下载地址：

```bash
链接：https://pan.baidu.com/s/1UI9H2DDqnHdYslkvAFHCpA 
提取码：jtfe
```

文件下载后解压到：
```bash
EuroTruckSelfDriver/selfDriverInEuroTruck/Road/model/
路径下替换已有的2和test文件夹。
```

完整工程下载到本地后，在pycharm中打开工程，右键运行

```bash
EuroTruckSelfDriver/selfDriverInEuroTruck/infer.py
即可推理视屏流的欧卡2游戏画面
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200830233202401.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NjY4NDM2,size_16,color_FFFFFF,t_70#pic_center)
# 核心代码
在infer.py的122行：
```bash
 # 预测结果
 output_im = PILImage.fromarray(np.asarray(parsing, dtype=np.uint8))
 output_im.putpalette(palette)
 arr = np.asarray(output_im)
 # img2infer = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) # 用来推理的图像
 # 推理的代码

# 城市景物分割做了十多个类别，从0-19不同的值代表不同的含义，做一下处理为了可视化！
 arr = np.where(arr == 13, 125, arr) # 将车子的像素改成125（灰色）
 arr = np.where(arr == 0, 255, arr) # 将地面的像素改成0（白色）
 arr = np.where(arr < 30, 30, arr)# 将其他的像素都改成30（黑色）
```
