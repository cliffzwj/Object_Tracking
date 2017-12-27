# Object_Tracking
a usage on tensorflow object detection API

本例程采用win10 + py3.6 + tensorflow1.4.0 + opencv3的组合

1.使用tensorflow提供的object_detection APi创建的应用

2.采用opencv3提供实时object tracking功能，考虑到实时性采用ssd_mobilenet_v1_coco_2017_11_17模型

完善中......

2017.12.27
加入多线程，读取视频流，多进程，加载物体识别模型，用来加快实时性能既fps