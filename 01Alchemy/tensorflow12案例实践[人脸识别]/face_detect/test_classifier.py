# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/5/18 5:10 PM
# project   DeepLearingStudy

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2

import sys

import detect_face
import nn4 as network

from sklearn.externals import joblib

# face detection parameters
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor

# facenet embedding parameters

model_dir = '/mnt/hgfs/WorkSpace/data/detect_face/model_check_point/models.ckpt-500000'  # "Directory containing the graph definition and checkpoint files.")
image_size = 96  # "Image size (height, width) in pixels."
pool_type = 'MAX'  # "The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn = False  # "Enables Local Response Normalization after the first layers of the inception network."
seed = 42,  # "Random seed."
batch_size = None  # "Number of images to process in a batch."
frame_interval = 1  # frame intervals


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


if __name__ == '__main__':
    # 加载模型
    print('Creating networks and loading parameters')
    gpu_memory_fraction = 1.0
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, '/mnt/hgfs/WorkSpace/data/detect_face/model_check_point/')

    # 建立 facenet models
    print('建立facenet embedding模型')
    tf.Graph().as_default()
    sess = tf.Session()
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           image_size,
                                                           image_size, 3), name='input')

    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    embeddings = network.inference(images_placeholder, pool_type,
                                   use_lrn,
                                   1.0,
                                   phase_train=phase_train_placeholder)

    ema = tf.train.ExponentialMovingAverage(1.0)
    saver = tf.train.Saver(ema.variables_to_restore())

    model_checkpoint_path = '/mnt/hgfs/WorkSpace/data/detect_face/model_check_point/models-20160506.ckpt-500000'

    saver.restore(sess, model_checkpoint_path)
    print('facenet embedding模型建立完毕')

    # restore pre-trained knn classifier
    model = joblib.load('/mnt/hgfs/WorkSpace/data/detect_face/model_check_point/knn_classifier_gender.models')
    print('knn classifier loaded 建立完毕')

    # 读取一张图片
    image = cv2.imread(sys.argv[1])

    find_results = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray.ndim == 2:
        img = to_rgb(gray)

    bounding_boxes, points = detect_face.detect_face( img , minsize, pnet, rnet, onet, threshold, factor)

    nrof_faces = bounding_boxes.shape[0]  # number of faces

    for face_position in bounding_boxes:
        face_position = face_position.astype(int)

        cv2.rectangle(image, (face_position[0],
                              face_position[1]),
                      (face_position[2], face_position[3]),
                      (0, 255, 0), 2)

        crop = img[face_position[1]:face_position[3], face_position[0]:face_position[2], ]

        # 重新进行数据编排
        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)

        data = crop.reshape(-1, 96, 96, 3)

        emb_data = sess.run([embeddings],
                            feed_dict={images_placeholder: np.array(data),
                                       phase_train_placeholder: False})[0]

        # 利用KNN模型进行预测
        predict = model.predict(emb_data)

        print(predict)

        if predict[0] == 0:
            result = 'female'
        elif predict[0] == 1:
            result = 'male'

        cv2.putText(image, result, (face_position[0] - 10, face_position[1] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                    (255, 0, 0), thickness=2, lineType=2)

    # show result
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Show Result", image)
    cv2.waitKey(0)
