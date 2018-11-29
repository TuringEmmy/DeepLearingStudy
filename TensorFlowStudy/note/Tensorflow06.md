### 一、图像基本知识分析

**图像概念**

在数字化表示图片的时候，有三个因素。分别是图片的长、图片的宽、图片的颜色通道数。那么黑白图片的颜色通道数为1，它只需要一个数字就可以表示一个像素位；而彩色照片就不一样了，它有三个颜色通道，分别为RGB，通过三个数字表示一个像素位。TensorFlow支持JPG、PNG图像格式，RGB、RGBA颜色空间。图像用与图像尺寸相同(height*width*chnanel)张量表示。图像所有像素存在磁盘文件，需要被加载到内存。

**图像大小压缩**

预处理阶段完成图像操作，缩小、裁剪、缩放、灰度调整等。图像加载后，翻转、扭曲，使输入网络训练信息多样化，缓解过拟合。Python图像处理框架PIL、OpenCV。TensorFlow提供部分图像处理方法。

```python
tf.image.resize_images 压缩图片导致定大小
```

### 二、图片文件读取

> 同样图像加载与二进制文件相同。图像需要解码。输入生成器(tf.train.string_input_producer)找到所需文件，加载到队列。**tf.WholeFileReader** 加载完整图像文件到内存，**WholeFileReader.read** 读取图像，**tf.image.decode_jpeg** 解码JPEG格式图像。图像是三阶张量。RGB值是一阶张量。加载图像格 式为[batch_size,image_height,image_width,channels]。批数据图像过大过多，占用内存过高，系统会停止响应。**直接加载TFRecord文件，可以节省训练时间。支持写入多个样本。**

**管道读端多文件内容**

read只返回一个图片的值。所以我们在之前处理文件的整个流程中，后面的内容队列的出队列需要用特定函数去获取。

- **tf.train.batch** 读取指定大小（个数）的张量
- **tf.train.shuffle_batch** 乱序读取指定大小（个数）的张量

### 三、二进制文件读取分析

1. 构造文件队列
2. 构造二进制文件读取器，读取内容
3. 解码内容，二进制文件解码
4. 分隔出图片和标签，切出特征值和目标值
5. 可以对图片的特征数据进行形状的改变
6. 批处理数据

### 四、二进制文件读取

```python
FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("./CIFAR/cifar-10-batches-bin/")
tf.app.flags.DEFINE_string("cifar_dir", "./CIFAR/data.bin")

class CifarRead(object):
    def __init__(self, filelist):
        self.filelist = filelist 
        self.height = 32
        self.width = 32
        self.channel = 3
        self.label_bytes = 1 
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def read_decode(self):
        file_queue = tf.train.string_input_producer(self.filelist)
        reader = tf.FixedLengthRecordReader(self.bytes)
        key, value = reader.read((file_queue))  
        label_image = tf.decode_raw(value, tf.uint8) 
        label = tf.cast(tf.slice(label_image, [0], [self.bytes]), tf.int32)
        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])        
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=10, capacity=12)

if __name__ == '__main__':
    file_name = os.listdir(FLAGS.cifar_dir)
    filelist = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]
    cf = CifarRead(filelist)
    image_batch, label_batch = cf.read_decode()
    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord=coord)
        coord.request_stop()
        coord.join()
```



### 五、tfrecords文件的读取与存储

```python
FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("./CIFAR/cifar-10-batches-bin/")
tf.app.flags.DEFINE_string("cifar_dir", "./CIFAR/data.bin", "filedir")
tf.app.flags.DEFINE_string("cifar_dir", "./CIFAR/temp.tfrecords", "save to tfrecords file")


class CifarRead(object):

    def __init__(self, filelist):
        self.filelist = filelist
        self.height = 32
        self.width = 32
        self.channel = 3
        self.label_bytes = 1 
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def write_tfrecords(self, image_batch, label_batch):
        writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)
        for i in range(10):
            image = image_batch[i].eval().tostring()
            label = label_batch[i].eval()[0]  
            example = tf.train.Example(features=tf.train.Features(feature={
                "iamge": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            writer.write(example.SerializeToString())
        writer.close()
        return None

    def read_tfrecords(self):
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])
        reader = tf.TFRecordReader()
        key, value = reader.read((file_queue))
        tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
        image = tf.decode_raw(features["image"], tf.uint8)
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        label = tf.cast(features["label"], tf.int32)
        image_batch,label_batch=tf.train.batch([image_reshape, label],batch_size=10,num_threads=3,capacity=10)
        return image_batch,label_batch


if __name__ == '__main__':
    file_name = os.listdir(FLAGS.cifar_dir)
    filelist = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]
    cf = CifarRead(filelist)
    image_batch, label_batch = cf.read_tfrecords()

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord=coord)
        coord.request_stop()
        coord.join()
```

