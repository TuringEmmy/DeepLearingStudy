# life is short, you need use python to create something!
# author    TuringEmmy
# time      18-11-22 下午3:51
# project   MachineLearning

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# csv文件读取：
# 1. 先找到文件，构造一个列表
# 2. 构造文件队列
# 3. 构造哦阅读器，读取队列内容(默认一行)
# 4. 解码内容
# 5. 批处理(多个样本)


def csvread(filelist):
    """
    读取CSV文件
    :param filelist: 文件路径+名字列表
    :return: 读取的内容
    """
    # 1. 构造文件的队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2. 构造csv阅读器
    reader = tf.TextLineReader()

    key, value = reader.read(file_queue)

    # print(value)

    # 3. 对每行内容进行解码
    # record_defaults:指定每一个样本每一列的类型，指定默认值
    records = [[1], [1], [1], [1]]
    col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=records)
    # print(col1, col2, col3, col4)

    # 4. 数据样本读取出来了，就可以进行训练了
    id_batch, city_batch, province_batch, cost_batch = tf.train.batch([col1, col2, col3, col4], batch_size=10,
                                                                      num_threads=6, capacity=9)

    print(id_batch, city_batch, province_batch, cost_batch)
    return id_batch, city_batch, province_batch, cost_batch


if __name__ == '__main__':
    # 找到文件，放入列表  路径+名字
    file_name = os.listdir("./csvfile/")
    # print(file_name)
    filelist = [os.path.join("./csvfile/", file) for file in file_name]

    id, city, province, cost = csvread(filelist=filelist)

    # 开启会话运行结果
    with tf.Session() as session:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()
        # 开启读文件的线程
        threads = tf.train.start_queue_runners(session, coord=coord)

        # 打印读取的内容
        print(session.run([id, city, province, cost]))

        print("*" * 100)

        # 不要忘记回收线程
        coord.request_stop()
        coord.join(threads=threads)

"""
batch_size：批处理大小，跟队列，数据的数量没有影响，只决定 这批次取多少
capacity：一次性能存多少个数据
"""
