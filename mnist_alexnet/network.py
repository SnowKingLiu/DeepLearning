# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/5/11 上午11:43
import tensorflow as tf


def my_conv2d(name, x, w, b, strides=1):
    """
    定义卷积操作
    :param name:
    :param x:
    :param w:
    :param b:
    :param strides:
    :return:
    """
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    # 使用relu激活函数
    return tf.nn.relu(x, name=name)


def my_max_pool2d(name, x, k=2):
    """
    定义池化层操作
    :param name:
    :param x:
    :param k:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def my_norm(name, l_input, l_size=4):
    """
    规范化操作
    :param name:
    :param l_input:
    :param l_size:
    :return:
    """
    return tf.nn.lrn(l_input, l_size, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)


# 定义整个网络
def alex_net(x, weights, biases, dropout):
    # Reshape输入图片
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 第一层卷积
    # 卷积
    conv1 = my_conv2d('conv1', x, weights['wc1'], biases['bc1'])
    # 亚采样
    pool1 = my_max_pool2d('pool1', conv1, k=2)
    # 标准化
    norm1 = my_norm('norm1', pool1, l_size=4)

    # 第二层卷积
    # 卷积
    conv2 = my_conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    # 亚采样
    pool2 = my_max_pool2d('pool2', conv2, k=2)
    # 标准化
    norm2 = my_norm('norm2', pool2, l_size=4)

    # 第三层卷积
    # 卷积
    conv3 = my_conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    # 亚采样
    pool3 = my_max_pool2d('pool3', conv3, k=2)
    # 标准化
    norm3 = my_norm('norm3', pool3, l_size=4)

    # 第四层卷积
    # 卷积
    conv4 = my_conv2d('conv4', norm3, weights['wc4'], biases['bc4'])

    # 第五层卷积
    # 卷积
    conv5 = my_conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    # 亚采样
    pool5 = my_max_pool2d('pool5', conv5, k=2)
    # 标准化
    norm5 = my_norm('norm4', pool5, l_size=4)

    # 全连接层1
    fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # 全连接层2
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # 输出层
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

