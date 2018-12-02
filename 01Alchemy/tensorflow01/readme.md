## 深度学习框架Tensorflow学习与应用

### 一、Anaconda安装

1. Window,MacOS,Linux都已支持Tensorflow。
2. Window用户只能使用python3.5(64bit)。MacOS,Linux支持python2.7和python3.3+。
3. 有GPU可以安装带GPU版本的，没有GPU就安装CPU版本的。

>  推荐安装Anaconda，pip版本大于8.1。



### 二、Tensorflow安装

Windows安装Tensorflow

- CPU版本：

管理员方式打开命令提示符，输入命令：pip install tensorflow

- GPU版本：

管理员方式打开命令提示符，输入命令：pip install tensorflow-gpu

- 更新Tensorflow：

`pip uninstall tensorflow`
`pip install tensorflow`

NOTE: TensorFlow requires MSVCP140.DLL, which may not be installed on your system. If,
when you import tensorflow as tf, you see an error about No module named
"_pywrap_tensorflow" and/or DLL load failed, check whether MSVCP140.DLL is in
your %PATH% and, if not, you should install the Visual C++ 2015 redistributable (x64
version)

Linux和MacOS安装Tensorflow

- CPU版本：

Python 2.7用户：`pip install tensorflow`
Python3.3+用户：`pip3 install tensorflow`

- GPU版本：

Python 2.7用户：`pip install tensorflow-gpu`
Python3.3+用户：`pip3 install tensorflow-gpu`





