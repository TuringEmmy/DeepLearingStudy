### 一、张量的阶和数据类型

TensorFlow用张量这种数据结构来表示所有的数据.一个张量有一个静态类型和动态类型的维数.张量可以在图中的节点之间流通.其实张量更代表的就是一种多位数组

| 数组array | 矩阵ndarray | 张量tensor |
| ------- | --------- | -------- |
|         |           |          |

**阶**

一个二阶张量就是我们平常所说的矩阵，一阶张量可以认为是一个向量.

| 阶    | 数学实例 | Python  | 例子                                       |
| ---- | ---- | ------- | ---------------------------------------- |
| 0    | 纯量   | (只有大小)  | s = 483                                  |
| 1    | 向量   | (大小和方向) | v = [1.1, 2.2, 3.3]                      |
| 2    | 矩阵   | (数据表)   | m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]    |
| 3    | 3阶张量 | (数据立体)  | t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]] |
| n    | n阶   | (自己想想看) | ....                                     |

**数据类型**

| 数据类型         | Python 类型    | 描述                         |
| ------------ | ------------ | -------------------------- |
| DT_FLOAT     | tf.float32   | 32 位浮点数.                   |
| DT_DOUBLE    | tf.float64   | 64 位浮点数.                   |
| DT_INT64     | tf.int64     | 64 位有符号整型.                 |
| DT_INT32     | tf.int32     | 32 位有符号整型.                 |
| DT_INT16     | tf.int16     | 16 位有符号整型.                 |
| DT_INT8      | tf.int8      | 8 位有符号整型.                  |
| DT_UINT8     | tf.uint8     | 8 位无符号整型.                  |
| DT_STRING    | tf.string    | 可变长度的字节数组.每一个张量元素都是一个字节数组. |
| DT_BOOL      | tf.bool      | 布尔型.                       |
| DT_COMPLEX64 | tf.complex64 | 由两个32位浮点数组成的复数:实数和虚数.      |
| DT_QINT32    | tf.qint32    | 用于量化Ops的32位有符号整型.          |
| DT_QINT8     | tf.qint8     | 用于量化Ops的8位有符号整型.           |
| DT_QUINT8    | tf.quint8    | 用于量化Ops的8位无符号整型.           |

### 二、张量的操作

**生成张量**

| 函数            | 用法                                       | 解释          |
| ------------- | ---------------------------------------- | ----------- |
| **zeros**     | **zeros(shape, dtype=tf.float32, name=None)** | 所有元素为零的张量   |
|               | **zeros_like(tensor, dtype=None, name=None)** | 给tensor定单张量 |
| **ones_like** | **ones_like(tensor, dtype=None, name=None)** | 给tensor定单张量 |
| **fill**      | **fill(dims, value, name=None)**         | 填充了标量值的张量   |
| **constant**  | **constant(value, dtype=None, shape=None, name='Const')** | 常数张量        |

```python
t1 = tf.constant([1, 2, 3, 4, 5, 6, 7])
t2 = tf.constant(-1.0, shape=[2, 3])
print(t1,t2)
```

张量--->`名字`+`形状`+`类型`

**创建张量**

| 函数                   | API                                      |                |
| -------------------- | ---------------------------------------- | -------------- |
| **truncated_normal** | **truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)** | 从截断的正态分布中输出随机值 |
| **random_normal**    | **random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)** | 从正态分布中输出随机值    |
| **random_uniform**   | **random_uniform(shape, minval=0.0, maxval=1.0, dtype=tf.float32, seed=None, name=None)** | 从均匀分布输出随机值     |
| **random_shuffle**   | **random_shuffle(value, seed=None, name=None)** | 沿其第一维度随机打乱     |
| **set_random_seed**  | **set_random_seed(seed)**                | 设置图级随机种子       |

```python
# 正态分布的 4X4X4 三维矩阵，平均值 0， 标准差 1
normal = tf.truncated_normal([4, 4, 4], mean=0.0, stddev=1.0)
```

**张良的变换**

- tf.string_to_number(string_tensor, out_type=None, name=None)
- tf.to_double(x, name='ToDouble')
- tf.to_float(x, name='ToFloat')
- tf.to_bfloat16(x, name='ToBFloat16')
- tf.to_int32(x, name='ToInt32')
- tf.to_int64(x, name='ToInt64')
- tf.cast(x, dtype, name=None)

**tf.string_to_number(string_tensor, out_type=None, name=None)**

将输入Tensor中的每个字符串转换为指定的数字类型。注意，int32溢出导致错误，而浮点溢出导致舍入值

```python
n1 = tf.constant(["1234","6789"])
n2 = tf.string_to_number(n1,out_type=tf.types.float32)

sess = tf.Session()

result = sess.run(n2)
print result

sess.close()
```

**形状和变换**

- tf.shape(input, name=None)
- tf.size(input, name=None)
- tf.rank(input, name=None)
- tf.reshape(tensor, shape, name=None)
- tf.squeeze(input, squeeze_dims=None, name=None)
- tf.expand_dims(input, dim, name=None)

**tf.shape(input, name=None)**

返回张量的形状。

```
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
shape(t) -> [2, 2, 3]
```

**静态形状与动态形状**

**静态维度** 是指当你在创建一个张量或者由操作推导出一个张量时，这个张量的维度是确定的。它是一个元祖或者列表。

**动态形状** 当你在运行你的图时，动态形状才是真正用到的。这种形状是一种描述原始张量在执行过程中的一种张量。如果你定义了一个没有标明具体维度的占位符，即用None表示维度，那么当你将值输入到占位符时，这些无维度就是一个具体的值，并且任何一个依赖这个占位符的变量，都将使用这个值。

| API                                      | exp               |
| ---------------------------------------- | ----------------- |
| **squeeze(input, squeeze_dims=None, name=None)** | 将input中维度是1的那一维去掉 |
| **expand_dims(input, dim, name=None)**   | 添加一个指定维度          |

**切片与扩展**

- tf.slice(input_, begin, size, name=None)
- tf.split(split_dim, num_split, value, name='split')
- tf.tile(input, multiples, name=None)
- tf.pad(input, paddings, name=None)
- tf.concat(concat_dim, values, name='concat')
- tf.pack(values, name='pack')
- tf.unpack(value, num=None, name='unpack')
- tf.reverse_sequence(input, seq_lengths, seq_dim, name=None)
- tf.reverse(tensor, dims, name=None)
- tf.transpose(a, perm=None, name='transpose')
- tf.gather(params, indices, name=None)
- tf.dynamic_partition(data, partitions, num_partitions, name=None)
- tf.dynamic_stitch(indices, data, name=None)

**其他**

| 作用      | API                                      |
| ------- | ---------------------------------------- |
| 张量复制与组合 | `identity(input, name=None)`             |
|         | `tuple(tensors, name=None, control_inputs=None)` |
|         | `group(*inputs, \**kwargs)`              |
|         | `no_op(name=None)`                       |
|         | `count_up_to(ref, limit, name=None)`     |
| 逻辑运算符   | ` logical_and(x, y, name=None)`          |
|         | `logical_not(x, name=None)`              |
|         | `logical_or(x, y, name=None)`            |
|         | `logical_xor(x, y, name='LogicalXor')`   |
| 比较运算符   | ` equal(x, y, name=None)`                |
|         | `not_equal(x, y, name=None)`             |
|         | `less(x, y, name=None) `                 |
|         | `less_equal(x, y, name=None)`            |
|         | `greater(x, y, name=None) `              |
|         | `greater_equal(x, y, name=None)`         |
|         | `select(condition, t, e, name=None)`     |
|         | `where(input, name=None)`                |
| 判断检查    | `is_finite(x, name=None)`                |
|         | `is_inf(x, name=None)`                   |
|         | `is_nan(x, name=None)`                   |
|         | `verify_tensor_all_finite(t, msg, name=None)` 断言张量不包含任何NaN或Inf |
|         | `check_numerics(tensor, message, name=None)` |
|         | `add_check_numerics_ops()`               |
|         | ` Assert(condition, data, summarize=None, name=None)` |
|         | `Print(input_, data, message=None, first_n=None, summarize=None, name=None)` |