# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/22 12:10
# project   MachineLearning

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/turing',
                           """数据集目录""")
tf.app.flags.DEFINE_integer('max_steps', 2000,
                            """训练次数""")
tf.app.flags.DEFINE_string('summary_dir', '/turing',
                           """事件文件目录""")


def main(argv):
    print(FLAGS.data_dir)
    print(FLAGS.max_steps)
    print(FLAGS.summary_dir)
    print(argv)


if __name__=="__main__":
    tf.app.run()