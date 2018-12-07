# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/7/18 1:07 PM
# project   DeepLearingStudy

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def rnn_model(model='lstm', input_data, output_data, vocal_size, run_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
    end_points = {}

    cell_fun = tf.contrib.rnn.BasicLSTMCell
    cell = tf.cell_fun(run_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    intial_state = cell.zero_stat(batch_size, tf.float32)
    embedding = tf.Variable(tf.random_uniform([vocal_size + 1, run_size], -1.0, 1.0))

    # 去找他属于那个词
    inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=intial_state)

    output = tf.reshape(outputs, [-1, run_size])

    weights=tf.Variable(tf.truncated_normal([run_size,vocal_size+1]) )

    bias=tf.Variable(tf.zeros(shape=[vocal_size+1]))

    logits=tf.nn.bias_add(tf.matmul(output,weights),bias=bias)

    labels=tf.one_hot(tf.reshape(output_data,[-1]),depth=vocal_size+1)

    # 进行交叉验证
    loss=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)

    total_loss = tf.reduce_mean(loss)

    train_op=tf.train.AdadeltaOptimizer(learning_rate).minimize(total_loss)

