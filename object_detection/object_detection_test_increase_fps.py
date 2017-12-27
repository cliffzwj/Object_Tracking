from multiprocessing import Queue, Pool

import cv2
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import multiprocessing

from object_detection.webvideo import WebcamVideoStream

sys.path.append("..")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 物体识别神经网络，向前传播获得识别结果
def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=5)
    return image_np


# 参数配置
class configs(object):
    def __init__(self):
        self.num_workers = 2  # worker数量
        self.queue_size = 5  # 多进程，输入输出，队列长度
        self.video_source = 0  # 0代表从摄像头读取视频流
        self.width = 1024  # 图片宽
        self.height = 800  # 图片高


args = configs()


# 定义用于多进程执行的函数word，每个进程执行work函数，都会加载一次模型
def worker(input_q, output_q):
    detection_graph = tf.Graph()
    with detection_graph.as_default():  # 加载模型
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    while True:  # 全局变量input_q与output_q定义，请看下文
        frame = input_q.get()  # 从多进程输入队列，取值
        output_q.put(detect_objects(frame, sess, detection_graph))  # detect_objects函数 返回一张图片，标记所有被发现的物品
    sess.close()


if __name__ == '__main__':

    input_q = Queue(maxsize=args.queue_size)  # 多进程输入队列
    output_q = Queue(maxsize=args.queue_size)  # 多进程输出队列
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()

    while True:
        frame = video_capture.read()  # video_capture多线程读取视频流
        input_q.put(frame)  # 视频帧放入多进程输入队列
        frame = output_q.get()  # 多进程输出队列取出标记好物体的图片

        cv2.imshow('Video', frame)  # 展示已标记物体的图片
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pool.terminate()  # 关闭多进程
    video_capture.stop()  # 关闭视频流
    cv2.destroyAllWindows()  # opencv窗口关闭
