# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/5/22 上午10:00
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 加载数据
mnist = input_data.read_data_sets("../mnist_alexnet/mnist_data", one_hot=True)

# 构建回归模型
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, w) + b

# 定义损失函数
# 输入真实值的占位符
y_ = tf.placeholder(tf.float32, [None, 10])

# 用tf.nn.softmax_cross_entropy_with_logits来计算预测值y和真实值y_的差值，并取均值
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# 采用SGD作为优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


