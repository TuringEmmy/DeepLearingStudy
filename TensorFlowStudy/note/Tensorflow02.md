### 一、TensorFlower图的结构

Tensorflow有一下几个简单的步骤：

- 使用 tensor 表示数据.
- 使用图 (graph) 来表示计算任务.
- 在会话（session)中运行图s

**图**

TensorFlow程序通常被组织成一个构建阶段和一个执行阶段. 在构建阶段, op的执行步骤被描述成一个图. 在执行阶段, 使用会话执行执行图中的op。

**构建图**

```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)
```

最终的打印声明生成

```python
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
```

构建图的第一步, 是创建源 op (source op). 源 op 不需要任何输入, 例如 常量 (Constant). 源 op 的输出被传递给其它 op 做运算.**TensorFlow Python 库有一个默认图 (default graph), op 构造器可以为其增加节点. 这个默认图对 许多程序来说已经足够用了.**

默认Graph值始终注册，并可通过调用访问 tf.get_default_graph()

```python
import tensorflow as tf

# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点,加到默认图中.构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)

print tf.get_default_graph(),matrix1.graph,matrix2.graph
```

**重要注意事项：此类对于图形构造不是线程安全的。所有操作都应从单个线程创建，或者必须提供外部同步。除非另有说明，所有方法都不是线程安全的**

**在会话中启动图**

启动图的第一步是创建一个Session对象，如果无任何创建参数，会话构造器将启动默认图。

调用Session的run()方法来执行矩阵乘法op, 传入product作为该方法的参数，会话负责传递op所需的全部输入，op通常是并发执行的。

```python
# 启动默认图.
sess = tf.Session()

# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.返回值 'result' 是一个 numpy `ndarray` 对象.
result = sess.run(product)
print result

# 任务完成, 关闭会话.
sess.close()
```

> Session对象在使用完后需要关闭以释放资源，当然也可以使用上下文管理器来完成自动关闭动作。

**OP**

计算图中的每个节点可以有任意多个输入和任意多个输出，每个节点描述了一种运算操作（operation, op），节点可以算作运算操作的实例化（instance）。一种运算操作代表了一种类型的抽象运算，比如矩阵乘法、加法。tensorflow内建了很多种运算操作，如下表所示：

| 类型      | 示例                                       |
| ------- | ---------------------------------------- |
| 标量运算    | Add、Sub、Mul、Div、Exp、Log、Greater、Less、Equal |
| 向量运算    | Concat、Slice、Splot、Constant、Rank、Shape、Shuffle |
| 矩阵运算    | Matmul、MatrixInverse、MatrixDeterminant   |
| 带状态的运算  | Variable、Assign、AssignAdd                |
| 神经网络组件  | SoftMax、Sigmoid、ReLU、Convolution2D、MaxPooling |
| 存储、恢复   | Save、Restore                             |
| 队列及同步运算 | Enqueue、Dequeue、MutexAcquire、MutexRelease |
| 控制流     | Merge、Switch、Enter、Leave、NextIteration   |

**feed**

临时替代图中的任意操作中的tensor可以对图中任何操作提交补丁,直接插入一个 tensor。feed 使用一个 tensor 值临时替换一个操作的输入参数，从而替换原来的输出结果

### 二、图

**tf.Graph**

一个图包含一组表示 tf.Operation计算单位的对象和tf.Tensor表示操作之间流动的数据单元的对象。默认Graph值始终注册，并可通过调用访问 tf.get_default_graph。

```python
a = tf.constant(1.0)
assert c.graph is tf.get_default_graph()
```

**图的其它属性和方法**

图是一个类，当然会有它自己属性和方法

`as_default()`返回一个上下文管理器，使其成为Graph默认图形。

如果要在同一过程中创建多个图形，则应使用此方法。为了方便起见，提供了一个全局默认图形，如果不明确地创建一个新的图形，所有操作都将添加到此图形中。使用该with关键字的方法来指定在块的范围内创建的操作应添加到此图形中。

```python
g = tf.Graph()
with g.as_default():
  a = tf.constant(1.0)
  assert c.graph is g
```

### 三、会话

`tf.Session`运行TensorFlow操作图的类，一个包含ops执行和tensor被评估, 注意使用Session的时候可以自动给关闭哦

```python
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
sess = tf.Session()
print(sess.run(c))
```

**在开启会话的时候指定图**

```
with tf.Session(graph=g) as sess:
```

**释放资源**会话拥有很多的资源，在不需要的时候，要及时释放出去

```
# 使用close手动关闭
sess = tf.Session()
sess.run(...)
sess.close()

# 使用上下文管理器
with tf.Session() as sess:
  sess.run(...)
```

### 四、会话的run方法

`run(fetches, feed_dict=None, options=None, run_metadata=None)`

运行ops和计算tensor

- fetches 可以是单个图形元素，或任意嵌套列表，元组，namedtuple，dict或OrderedDict
- feed_dict 允许调用者覆盖图中指定张量的值

如果a,b是其它的类型，比如tensor，同样可以覆盖原先的值

```
a = tf.placeholder(tf.float32, shape=[])
b = tf.placeholder(tf.float32, shape=[])
c = tf.constant([1,2,3])

with tf.Session() as sess:
    a,b,c = sess.run([a,b,c],feed_dict={a: 1, b: 2,c:[4,5,6]})
    print(a,b,c)

```

**错误**

- RuntimeError：如果它Session处于无效状态（例如已关闭）。
- TypeError：如果fetches或feed_dict键是不合适的类型。
- ValueError：如果fetches或feed_dict键无效或引用 Tensor不存在。

### 五、其它属性和方法

**graph**

返回本次会话中的图

**as_default()**

返回使此对象成为默认会话的上下文管理器。

获取当前的默认会话，请使用 tf.get_default_session

```
c = tf.constant(..)
sess = tf.Session()

with sess.as_default():
  assert tf.get_default_session() is sess
  print(c.eval())

```

**注意：** 使用这个上下文管理器并不会在退出的时候关闭会话，还需要手动的去关闭

```
c = tf.constant(...)
sess = tf.Session()
with sess.as_default():
  print(c.eval())
# ...
with sess.as_default():
  print(c.eval())

sess.close()
```