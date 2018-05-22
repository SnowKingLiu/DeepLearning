# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/5/11 上午11:36
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("mnist_data", one_hot=True)

# 定义网络超参数
learning_rate = 0.001
training_iters = 200000
batch_size = 200
display_step = 10

# 定义网络的参数
# 输入的维度是28*28
n_input = 784
# 标记的维度是0-9
n_classes = 10
# Dropout的概率是0.75
dropout = 0.75

# 输入占位符
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32)


def my_conv2d(name, my_x, w, b, strides=1):
    """
    定义卷积操作
    :param name:
    :param my_x:
    :param w:
    :param b:
    :param strides:
    :return:
    """
    my_x = tf.nn.conv2d(my_x, w, strides=[1, strides, strides, 1], padding='SAME')
    my_x = tf.nn.bias_add(my_x, b)
    # 使用relu激活函数
    return tf.nn.relu(my_x, name=name)


def my_max_pool2d(name, my_x, k=2):
    """
    定义池化层操作
    :param name:
    :param my_x:
    :param k:
    :return:
    """
    return tf.nn.max_pool(my_x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def my_norm(name, l_input, l_size=4):
    """
    规范化操作
    :param name:
    :param l_input:
    :param l_size:
    :return:
    """
    return tf.nn.lrn(l_input, l_size, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)


# 定义所有的网络参数
weights = {
    "wc1": tf.Variable(tf.random_normal([11, 11, 1, 96])),
    "wc2": tf.Variable(tf.random_normal([5, 5, 96, 256])),
    "wc3": tf.Variable(tf.random_normal([3, 3, 256, 384])),
    "wc4": tf.Variable(tf.random_normal([3, 3, 384, 384])),
    "wc5": tf.Variable(tf.random_normal([3, 3, 384, 256])),
    "wd1": tf.Variable(tf.random_normal([2*2*256, 4096])),
    "wd2": tf.Variable(tf.random_normal([4096, 4096])),
    "out": tf.Variable(tf.random_normal([4096, 10])),
}

biases = {
    "bc1": tf.Variable(tf.random_normal([96])),
    "bc2": tf.Variable(tf.random_normal([256])),
    "bc3": tf.Variable(tf.random_normal([384])),
    "bc4": tf.Variable(tf.random_normal([384])),
    "bc5": tf.Variable(tf.random_normal([256])),
    "bd1": tf.Variable(tf.random_normal([4096])),
    "bd2": tf.Variable(tf.random_normal([4096])),
    "out": tf.Variable(tf.random_normal([n_classes])),
}
# weights = {
#     "wc1": tf.Variable(tf.random_normal([11, 11, 1, 96])),
#     "wc2": tf.Variable(tf.random_normal([5, 5, 96, 256])),
#     "wc3": tf.Variable(tf.random_normal([3, 3, 256, 384])),
#     "wc4": tf.Variable(tf.random_normal([3, 3, 384, 384])),
#     "wc5": tf.Variable(tf.random_normal([3, 3, 384, 256])),
#     "wd1": tf.Variable(tf.random_normal([4*4*256, 4096])),
#     "wd2": tf.Variable(tf.random_normal([4096, 4096])),
#     "out": tf.Variable(tf.random_normal([4096, 10])),
# }
#
# biases = {
#     "bc1": tf.Variable(tf.random_normal([96])),
#     "bc2": tf.Variable(tf.random_normal([256])),
#     "bc3": tf.Variable(tf.random_normal([384])),
#     "bc4": tf.Variable(tf.random_normal([384])),
#     "bc5": tf.Variable(tf.random_normal([256])),
#     "bd1": tf.Variable(tf.random_normal([4096])),
#     "bd2": tf.Variable(tf.random_normal([4096])),
#     "out": tf.Variable(tf.random_normal([n_classes])),
# }


# 定义整个网络

def alex_net(alex_x, alex_weights, alex_biases, alex_dropout):
    # Reshape输入图片
    alex_x = tf.reshape(alex_x, shape=[-1, 28, 28, 1])

    # 第一层卷积
    # 卷积
    conv1 = my_conv2d('conv1', alex_x, alex_weights['wc1'], alex_biases['bc1'])
    # 亚采样
    pool1 = my_max_pool2d('pool1', conv1, k=2)
    # 标准化
    norm1 = my_norm('norm1', pool1, l_size=4)

    # 第二层卷积
    # 卷积
    conv2 = my_conv2d('conv2', norm1, alex_weights['wc2'], alex_biases['bc2'])
    # 亚采样
    pool2 = my_max_pool2d('pool2', conv2, k=2)
    # 标准化
    norm2 = my_norm('norm2', pool2, l_size=4)

    # 第三层卷积
    # 卷积
    conv3 = my_conv2d('conv3', norm2, alex_weights['wc3'], alex_biases['bc3'])
    # 亚采样
    pool3 = my_max_pool2d('pool3', conv3, k=2)
    # 标准化
    norm3 = my_norm('norm3', pool3, l_size=4)

    # 第四层卷积
    # 卷积
    conv4 = my_conv2d('conv4', norm3, alex_weights['wc4'], alex_biases['bc4'])

    # 第五层卷积
    # 卷积
    conv5 = my_conv2d('conv5', conv4, alex_weights['wc5'], alex_biases['bc5'])
    # 亚采样
    pool5 = my_max_pool2d('pool5', conv5, k=2)
    # 标准化
    norm5 = my_norm('norm5', pool5, l_size=4)

    # 全连接层1
    fc1 = tf.reshape(norm5, [-1, alex_weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, alex_weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1 = tf.nn.dropout(fc1, alex_dropout)

    # 全连接层2
    fc2 = tf.reshape(fc1, [-1, alex_weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, alex_weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2 = tf.nn.dropout(fc2, alex_dropout)

    # 输出层
    out = tf.add(tf.matmul(fc2, alex_weights['out']), alex_biases['out'])
    return out


# 构建模型，定义损失函数和优化器，构建评估函数
# 构建模型
pred = alex_net(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估函数
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 训练模型和评估模型
# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    # 开始训练，直到达到training_iters，即200000
    while step*batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # 计算损失值和准精度，输出
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print("Iter {}, Minibatch Loss={}, Training Accuracy={}".format(
                step*batch_size, "{:.6f}".format(loss), "{:.5f}".format(acc)))
        step += 1
    print("Optimization Finished!")

    # 计算测试集的精确度
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                                             y: mnist.test.labels[:256],
                                                             keep_prob: 1.}))
