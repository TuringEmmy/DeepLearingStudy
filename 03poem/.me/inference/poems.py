# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/5/18 10:02 PM
# project   DeepLearingStudy

import os

import tensorflow as tf
from inference.poems2 import generate_batch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from practice.models import rnn_model

from practice.dataset.poems import process_poems

FLAGS = tf.app.flags.FLAGS

# 批处理的大小
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size = ?')
# 学习率的大小
tf.app.flags.DEFINE_float('learing_rate', 0.01, 'learning_rate')
# 模型保存路径
tf.app.flags.DEFINE_string('check_point_dir', '/mnt/hgfs/WorkSpace/data/poem/model/', 'check_points_dir')
# 数据源的位置
tf.app.flags.DEFINE_string('file_path', '/mnt/hgfs/WorkSpace/data/poem/data/.txt', 'file_path')
# 模型的
tf.app.flags.DEFINE_integer('epoch', 50, 'check_points_dir')

start_token = "G"  # 起始
end_token = "E"  # 终止


def run_training():
    # 将一个词转换成向量，之后要做成映射，对应数字，并返回所有的语料库，参数为数据源
    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)

    # 生成batch
    batch_inputs, batch_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    # int32是因为把词映射成数字,
    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    # 因为有交叉熵
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    # 定义模型
    end_points = rnn_model(model='lstm', input=input_data, output_data=output_targets, vocab_size=len(vocabularies)
                           , run_size=128, num_layers=2, batch_size=64, learning_rate=0.01)

    # 模型的保存时候使用
    saver = tf.train.Saver(tf.global_variables_initializer())

    init_op = tf.train.Saver(tf.global_variables_initializer())
    with tf.Session as sess:
        sess.run(init_op)

        start_epoch = 0
        # 可能之前保存过，进行迁移需诶，接着干
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)

        if checkpoint:
            saver.restore(sess, checkpoint)
            # 从上一次的基础上进行
            start_epoch += int(checkpoint.split('-')[-1])

        print('[INFO] start training...')

        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(poems_vector) // FLAGS.batch_size

                # 计算完成后的loss值
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([end_points['total_loss'],
                                           end_points['last_state'],
                                           end_points['train_op']], feed_dict={
                        input_data: batch_inputs[n],
                        output_targets: batch_outputs[n]
                    })
                    n += 1
                    print('[INFO] Epoch: %d, batch: %d, train loss: %.6f' % (epoch, batch, loss))

                if epoch % 6 == 0:
                    saver.save(sess, '/mnt/hgfs/WorkSpace/data/poem/model/', global_step=epoch)
        except KeyboardInterrupt:
            print('[INFO] INterrupt manually, try saving heckpoin for now...')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
            print('[INFO] last epoch saved, next time will start from epoch{}.'.format(epoch))


def main(is_train):
    # 如果训练
    if is_train:
        print("training")
        run_training()
    else:
        print("test")
        begin_word = input('word')


if __name__ == '__main__':
    tf.app.run()
