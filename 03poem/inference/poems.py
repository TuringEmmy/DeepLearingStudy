# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/5/18 10:02 PM
# project   DeepLearingStudy

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

from models.model import rnn_model

from dataset.poems import process_poems, generate_batch

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size = ?')
tf.app.flags.DEFINE_float('learing_rate', 0.01, 'learning_rate')
tf.app.flags.DEFINE_string('check_point_dir', '/mnt/hgfs/WorkSpace/data/poem/model/', 'check_points_dir')
tf.app.flags.DEFINE_string('file_path', '/mnt/hgfs/WorkSpace/data/poem/data/.txt', 'file_path')
tf.app.flags.DEFINE_integer('epoch', 50, 'check_points_dir')

start_token = "G"  # 起始
end_token = "E"  # 终止


def run_training():
    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    batch_inputs, batch_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input=input_data, output_data=output_targets, vocab_size=len(vocabularies)
                           , run_size=128, num_layers=2, batch_size=64, learning_rate=0.01)


def main(is_train):
    if is_train:
        print("training")
        run_training()
    else:
        print("test")
        begin_word = input('word')


if __name__ == '__main__':
    tf.app.run()
