
"""
    @Author: Junjie Jin
    @Date: 2022/5/9
    @Description: OSNet
        轻量级网络
"""

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


# 组卷积
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
        print(in_ch)
        res = []
        for _ in range(self.groups):
            res.append(self.convs[_](x[..., _ * in_ch:(_ + 1) * in_ch]))
        return tf.concat(res, axis=-1)


class Conv(layers.Layer):

    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, use_bias=False, groups=1):
        super(Conv, self).__init__()
        # self.conv = layers.Conv2D(out_channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=use_bias, groups=1)

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
        # self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelGate(layers.Layer):

    def __init__(self, in_channels, out_channels, num_gates = None,
                 return_gate=False, gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gate = return_gate
        self.global_avgpool = layers.GlobalAveragePooling2D()
        self.fc1 = Conv(in_channels, in_channels//reduction, kernel_size=1, use_bias=True)
        self.norm1 = None
        if layer_norm:
            self.norm1 = layers.LayerNormalization()
        self.relu = layers.ReLU()
        self.fc2 = Conv(in_channels//reduction, num_gates, kernel_size=1, use_bias=True)


        pass

    def call(self):
        return 0

class OSBlock(layers.Layer):

    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1)
        self.conv2a = LightConv(mid_channels, mid_channels)
        self.conv2b = models.Sequential([LightConv(mid_channels, mid_channels), LightConv(mid_channels, mid_channels)])
        self.conv2c = models.Sequential([LightConv(mid_channels, mid_channels), LightConv(mid_channels, mid_channels), LightConv(mid_channels, mid_channels)])
        self.conv2d = models.Sequential([LightConv(mid_channels, mid_channels), LightConv(mid_channels, mid_channels), LightConv(mid_channels, mid_channels), LightConv(mid_channels, mid_channels)])

        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv(in_channels, out_channels)

    def call(self, x):

        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return tf.nn.relu(out)

input = layers.Input(shape=(8, 8, 4))
output = Conv(4, 4)(input)
model = models.Model(input, output)
model.summary()

