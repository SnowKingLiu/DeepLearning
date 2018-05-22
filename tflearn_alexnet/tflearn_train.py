# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/5/14 下午2:28
import tflearn
import tflearn.datasets.oxflower17 as oxflower17

from tflearn_alexnet.tflearn_network import tflearn_model


def train():
    x, y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
    model = tflearn_model()
    model.fit(x, y, n_epoch=1000, validation_set=0.1, shuffle=True, show_metric=True, batch_size=512, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_oxflowers17')


if __name__ == '__main__':
    train()
