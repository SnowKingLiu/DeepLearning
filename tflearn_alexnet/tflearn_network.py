# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/5/14 下午2:30
import tflearn
from tflearn import input_data, conv_2d, max_pool_2d, local_response_normalization, fully_connected, dropout, regression


def tflearn_model():
    network = input_data(shape=[None, 227, 227, 3])

    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 5, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 384, 3, strides=4, activation='relu')

    network = conv_2d(network, 384, 3, strides=4, activation='relu')

    network = conv_2d(network, 256, 3, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 17, activation='softmax')
    # 回归操作，同时指定网络所使用的学习率、损失函数和优化器
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)

    model = tflearn.DNN(network, checkpoint_path='model_alexnet', max_checkpoints=1, tensorboard_verbose=2)

    return model
