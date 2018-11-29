# life is short, you need use python to create something!
# author    TuringEmmy
# time      18-11-22 下午7:03
# project   MachineLearning
import os

import tensorflow as tf


def picture_read(filelist):
    """
    读取猫图片病转换成张量
    :param filelist: 文件路径+名字列表
    :return: 每张图片的张量
    """
    # 1. 构造文件队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2. 构造阅读器取读取图片内容(默认读取一张图片)
    reader = tf.WholeFileReader()

    key, value = reader.read(file_queue)

    print(value)

    # 3. 对读取的图片数据进行解码
    image = tf.image.decode_jpeg(value)
    print(image)

    # 4. 处理图片的大小(统一大小)
    image_resize = tf.image.resize_images(image, [500, 300])
    print(image_resize)
    print("*" * 100)
    # 一定要注意：图片提取之后，每连续的三站刚才是一张图片的值哦，
    # 因为在批处理的时候所有数据形状必须定义
    image_resize.set_shape([500, 300, 3])  # 固定形状

    # 5. 进行批处理
    image_batch = tf.train.batch([image_resize], batch_size=100, num_threads=12, capacity=50)
    print(image_batch)
    return image_batch


if __name__ == '__main__':
    # 1.找到文件 放入列表
    file_name = os.listdir("./cat/")

    filelist = [os.path.join("./cat/", file) for file in file_name]

    image_resize = picture_read(filelist)

    with tf.Session() as session:
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(session, coord=coord)

        print(session.run(image_resize))

        coord.request_stop()
        coord.join()
