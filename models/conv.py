
"""
    @Author: Junjie Jin
    @Date: 2022/5/10
    @Description: 实现常用卷积块
"""


import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import math


class GroupsConv(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, groups=1, use_bias=False):

        super(GroupsConv, self).__init__()
        assert out_channels % groups == 0, 'out_channels 不能 整除 groups'
        self.convs = []
        for _ in range(groups):
            self.convs.append(layers.Conv2D(out_channels // groups, kernel_size=kernel_size, strides=strides, padding='same', use_bias=use_bias))
        self.groups = groups
        self.single = in_channels // groups

    def call(self, x):

        in_ch = x.shape[-1] // self.groups
        res = []
        for _ in range(self.groups):
            res.append(self.convs[_](x[..., _ * in_ch:(_ + 1) * in_ch]))
        return tf.concat(res, axis=-1)


class Conv(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, use_bias=False, groups=1):
        super(Conv, self).__init__()
        self.conv = GroupsConv(in_channels, out_channels, kernel_size, strides, use_bias=use_bias, groups=groups)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 深度可分离卷积
class LightConv(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(LightConv, self).__init__()
        self.conv1 = GroupsConv(in_channels, out_channels, kernel_size=kernel_size, strides=1, use_bias=False,
                                groups=out_channels)
        self.conv2 = layers.Conv2D(out_channels, kernel_size=1, strides=1, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 幽灵卷积
class GhostConv(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, dw_size=3, strides=1, ratio=2):
        super(GhostConv, self).__init__()
        self.oup = out_channels
        init_channels = math.ceil(out_channels // ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = models.Sequential([
            layers.Conv2D(init_channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.cheap_operation = models.Sequential([
            GroupsConv(init_channels, new_channels, kernel_size=dw_size, strides=1, groups=init_channels),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = tf.concat([x1, x2], axis=-1)
        return out[..., :self.oup]


class Bottleneck(layers.Layer):
    def __init__(self, output_dim, strides=1, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.conv1 = layers.Conv2D(output_dim // 4, kernel_size=1, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_dim // 4, kernel_size=3, strides=strides, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(output_dim, kernel_size=1, padding="same", use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def build(self, input_shape):
        super(Bottleneck, self).build(input_shape)

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class ResBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, strides=1):
        super(ResBlock, self).__init__()

        self.left = models.Sequential([
            layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization()
        ])
        self.short_cut = models.Sequential()
        if strides != 1 or in_channels != out_channels:
            self.short_cut = models.Sequential([
                layers.Conv2D(out_channels, kernel_size=1, strides=1, use_bias=False),
                layers.BatchNormalization()
            ])

    def call(self, x):
        return tf.nn.relu(self.left(x) + self.short_cut(x))


