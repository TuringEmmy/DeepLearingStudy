**辨析机器学习与深度学习**

| 学习   | 算法                                | 领域      |
| ---- | --------------------------------- | ------- |
| 机器学习 | 分类:神经网络(简单),回归                    | 传统的预测问题 |
| 深度学习 | 神经网络(深度);图像:卷积神将网路;自然语言的处理：循环神将网络 | 图像处理    |



**TensorFlow特点**

| 特点                   | 详情                                       |
| -------------------- | ---------------------------------------- |
| 高度的灵活性               | TensorFlow 不是一个严格的“神经网络”库。只要你可以将你的计算表示为一个数据流图，你就可以使用Tensorflow。 |
| 真正的可移植性（Portability） | Tensorflow 在CPU和GPU上运行，比如说可以运行在台式机、服务器、手机移动设备等等。 |
| 多语言支持                | Tensorflow 有一个合理的c++使用界面，也有一个易用的python使用界面来构建和执行你的graphs。 |
| 性能最优化                | Tensorflow 给予了线程、队列、异步操作等以最佳的支持，Tensorflow 让你可以将你手边硬件的计算潜能全部发挥出来。 |



**安装TensorFlow**

运行速度		数据量：特征多，图片; 算法:设计本身比较复杂	计算		等待很长的时间去优化	几小时，几天

CPU不是专门为了 计算而设计的, 处理业务，计算能力不是特别突出

GPU:运行操作系统，专门为计算设计的

如果是虚拟机,ubuntu是虚拟的不能GPU

**选择类型**

TensorFlower仅支持CPU



**ubuntu/linux**

使用pip安装,分别有2.7和3.6版本的

```
# 仅使用 CPU 的版本
$  pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp27-none-linux_x86_64.whl

$  pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp36-cp36m-linux_x86_64.whl
```

**Mac**

macX下也可以安装2.7和3.4、3.5的CPU版本

```
# 2.7
$ pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py2-none-any.whl

# 3.4、3.5
$ pip3 install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
```

