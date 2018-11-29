# life is short, you need use python to create something!
# author    TuringEmmy
# time      11/26/18 7:19 PM
# project   DeepLearingStudy

# 全连接神经网络的缺点
# 1. 参数太多，在cifar-10的数据集中，只有32*32*3，就会有这么多权重，换成更大的图片，需要更多
# 2. 没有利用像素之间的位置信息，对于图像识别任务来说，每个像素与周围的像素联系紧密
# 3. 层数限制


# 神经网络的基本组成包括：输入层，隐藏曾，输出层
# 面试容易问到撒asda


# 这里有笔试题哦

# 输入体积大小
# H_1*W_1*D_1
#
# 四个超参数
# Filter数量K
# Filter大小F
# 步长S
#
# 输出体积：
#
# H_2=(H_1-F+2P)/S+1
# W_2=(W_1-F)/S+1
# D_2=K
