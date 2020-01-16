# person-detection
TensorRT person detection RFBNet300
### 首先声明，原始的pytorch程序是songwsx大佬的(https://github.com/songwsx/RFSong-7993.git)

### 大体思路:

SSD类的网络都是prior anchor和网络输出的anchor进行解码运算。为了便于运算对网络输出做了如下处理：

1. pytorch网络输出是(7759,4)和(7759,2)，为了便于运算，将其reshape成(1,7759*4*7759*2),所以网络的输出为(1,46554)
2. 对网络输出的数据和prior anchor 一起做解码运算
3. 图像数据输入网络，使用的是 cuda 的opencv读入gpu上，然后写个cuda程序做预处理送入网络，这么做的原因是在用cpu处理的时候速度有些慢
4. 网络输出的数据到转换到真实的box都在cpu上运行，包括后面的nms

### 系统环境：

​	1.ubuntu 18.04

​	2.CUDA  10.2

​	3.TensorRT 6.0

​	4.CPU i5-8250

	5. GPU MX250

  6. opencv 3.4.6(CUDA)

### 运行方法:

	1. 终端切换到程序根目录

   	2. cd cuda_lib && mkdir build && cmake .. && make
   	3. cd ../
   	4. mkdir build &&  cmake && make

### 注意事项：

1. 注意更改main.cpp里的你要运行的视频地址
2. 根据我的CMakeLists.txt 配置你的CMakeLists.txt
3. cuda 的opencv 正确安装

### 测试:

1.测试时间: 1280 × 720视频 从模型输入到显示视频为18ms左右，此测试在MX250移动显卡上。

### TODO:

将来想法是将其推理时间减小到10ms左右，主要是模型压缩，还有将解码部分全部有gpu上运行
