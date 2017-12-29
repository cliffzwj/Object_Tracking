# Object_Tracking
a usage on tensorflow object detection API

本例程采用win10 + py3.6 + tensorflow1.4.0 + opencv3的组合

1.使用tensorflow提供的object_detection APi创建的应用

2.采用opencv3提供实时object tracking功能，考虑到实时性采用ssd_mobilenet_v1_coco_2017_11_17模型

完善中......

2017.12.27

加入多线程，读取视频流，多进程，加载物体识别模型，用来加快实时性能既fps

2017.12.28

1.将tensorflow更新为1.4.0的GPU版本，使用的显卡为Geforce 940M（笔记本），相关参数作调整以达到最佳fps
2.将显卡的显存使用修改为按需，这样可以在采用多进程加载tf.sessions的时候，自动控制显存使用

2017.12.29

增加视频流采样间隔配置参数，用来调整需要处理的帧，从而使不同设备达到最佳FPS