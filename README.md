# person-detection
TensorRT person detection RFBNet300

首先声明，原始的pytorch程序是songwsx大佬的(https://github.com/songwsx/RFSong-7993.git)。

##大体思路:
SSD类的网络都是prior anchor和网络输出的anchor进行解码运算。为了便于运算对网络输出做了如下处理：1. pytorch网络输出是(7759,4)和(7759,2)，为了便于运算，将其reshape成(1,7759*4*7759*2),所以网络的输出为(1,46554)
