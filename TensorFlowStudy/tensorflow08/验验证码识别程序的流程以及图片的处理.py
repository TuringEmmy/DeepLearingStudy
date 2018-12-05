# life is short, you need use python to create something!
# author    TuringEmmy
# time      11/29/18 3:12 PM
# project   DeepLearingStudy

# 1. 处理数据  图片  ---一一对应------标签文件
#         [20,80,3]

# 2. 识别验证码
#     从tfrecods中读取数据,每一张图片有image,label
#         [100,20,80,3]  [100,4]    [[13,45,18,18],[19,16,15,89],...]
#     建立模型，直接读取数据输入到模型当中   全连接层
#             x=[100,20*80*3]
#             y_predict=[100,4*26]
#             w=[20*80*3,4*26]
#             bias=[4*26]
#     建立损失，softmax, 交叉熵损失
# 先把[100,4]转换成 one_hot编码  --->[100,4*26]

# 3. 梯度下降