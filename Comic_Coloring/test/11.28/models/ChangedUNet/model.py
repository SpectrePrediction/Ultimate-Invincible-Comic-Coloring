import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import MaxPooling2D, Conv2D, LeakyReLU
from tensorflow.keras import Model, Sequential

"""
Inspired by [1]、[2]、[3]
Thanks
The front part of this model comes from changed u-net
The posterior segment comes from changed ResNeXt


[1]: https://arxiv.org/abs/1808.03240
[2]: https://arxiv.org/abs/1902.06838
[3]: https://arxiv.org/abs/2006.13717
[4]: Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. 2017. Deep multi-scale
convolutional neural network for dynamic scene deblurring. In Proceedings of
the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Vol. 2.
[5]: Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, and Kaiming He. 2017. ´
Aggregated residual transformations for deep neural networks. In Computer
Vision and Paern Recognition (CVPR), 2017 IEEE Conference on. IEEE, 5987–5995
[6]: https://arxiv.org/abs/1611.05431
[ ]: 
[ ]: 
"""


class DoubleConvBlock(Model):

    def __init__(self, out_channel, mid_channel=None, leaky_relu_alpha=0.2):
        """
        unet:   (conv2d -> bn -> relu)*2
        change: (conv2d->leaky_relu)*2
        why no use bn?
            we did not use any normalization layer throughout our
        networks to keep the range fexibility for accurate colorizing. It
        also reduces the memory usage and computational cost
        Inspired by [4] [5]
        don't need in_channel?
        :param out_channel:
        :param mid_channel:
        :param leaky_relu_alpha:
        """
        super(DoubleConvBlock, self).__init__()
        # # test
        # self.conv = Conv2D(32, 3)  # out, kernel_size, strides, padding,.., activation
        # self.leaky_relu = LeakyReLU(0.2)

        mid_channel = mid_channel if mid_channel else out_channel

        self.double_conv_relu = Sequential([
            Conv2D(mid_channel, kernel_size=3, padding='same'),
            LeakyReLU(leaky_relu_alpha),
            Conv2D(out_channel, kernel_size=3, padding='same'),
            LeakyReLU(leaky_relu_alpha)
        ])

    # def build(self, input_shape):
    #     tf.print(input_shape)

    def call(self, x):
        x = self.double_conv_relu(x)
        return x


class DownCNNBlock(Model):

    def __init__(self, out_channel, block_num=3, mid_channel=None, leaky_relu_alpha=0.2):

        super(DownCNNBlock, self).__init__()
        mid_channel = mid_channel if mid_channel else out_channel
        self.Block_num = block_num

        # self.double_conv_relu = Sequential([
        #     Conv2D(mid_channel, kernel_size=3, padding='same'),
        #     LeakyReLU(leaky_relu_alpha),
        #     Conv2D(out_channel, kernel_size=3, strides=2, padding='same'),
        #     LeakyReLU(leaky_relu_alpha)
        # ])

        self.double_conv_relu_list = []
        for i in range(self.Block_num):
            self.double_conv_relu_list.append(
                Sequential([
                    Conv2D(mid_channel, kernel_size=3, padding='same'),
                    LeakyReLU(leaky_relu_alpha),
                    Conv2D(out_channel, kernel_size=3, strides=2, padding='same'),
                    LeakyReLU(leaky_relu_alpha)
                ])
            )

    def call(self, x):
        local_x = x
        for i in range(self.Block_num):
            x = self.double_conv_relu_list[i](local_x)
            local_x = x

        return x


class Down(Model):

    def __init__(self, out_channel, mid_channel=None,
                 leaky_relu_alpha=0.2, pool_size=2):
        """
        unet:   MaxPooling2D -> DoubleConvBlock
        change: DoubleConvBlock->Conv2D strides 2->LeakyReLU
        using Conv2d+LeakyReLU replace MaxPooling2D
        :param out_channel:
        :param mid_channel:
        :param leaky_relu_alpha:
        :param pool_size:
        """
        super(Down, self).__init__()
        # self.maxpool_and_double_conv_block = Sequential([
        #     MaxPooling2D(pool_size),
        #     DoubleConvBlock(out_channel, mid_channel, leaky_relu_alpha)
        # ])

        self.maxpool_and_double_conv_block = Sequential([
            DoubleConvBlock(out_channel, mid_channel, leaky_relu_alpha),
            Conv2D(out_channel, kernel_size=3, strides=pool_size, padding='same'),
            LeakyReLU(leaky_relu_alpha)
        ])

    def call(self, x):
        x = self.maxpool_and_double_conv_block(x)

        return x


class GroupConv2D(Model):
    def __init__(self, output_channels, kernel_size,
                 strides=(1, 1), padding='valid', data_format=None,
                 dilation_rate=(1, 1), activation=None, groups=1, use_bias=True):
        super(GroupConv2D, self).__init__()
        self.groups = groups
        self.check_group(output_channels)
        self.group_out_num = output_channels // groups

        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(
                Conv2D(filters=self.group_out_num, kernel_size=kernel_size, strides=strides,
                       padding=padding, data_format=data_format, dilation_rate=dilation_rate,
                       activation=activation, use_bias=use_bias))

    def call(self, inputs):
        feature_map_list = []
        group_in_num = inputs.shape[-1] // self.groups
        self.check_group(inputs.shape[-1])

        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i*group_in_num: (i + 1) * group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out

    def check_group(self, num_channel):

        if not num_channel % self.groups == 0:
            raise ValueError("The value of output_channels:{0} must be divisible by the value of groups{1}".format(
                num_channel, self.groups
            ))


class ResNeXtBlock(Model):
    """
    Inspired by [6]
    """
    def __init__(self, out_channel, groups=32, leaky_relu_alpha=0.2):
        """
        out_channel 应当等于 in_channel
        out_channel应当是>=2 * groups(因为实现的关系，所以需要*2)
        c版本
        strides is only one
        have other up sample
        :param out_channel:
        :param mid_channel:
        :param groups:
        """
        super(ResNeXtBlock, self).__init__()
        # # 使用版本2.1不支持groups参数
        # self.resnext_conv_block = Sequential([
        #     Conv2D(out_channel * 2, kernel_size=1, padding='same', use_bias=False),
        #     LeakyReLU(leaky_relu_alpha),
        #     Conv2D(out_channel * 2, kernel_size=3, use_bias=False, padding='same', groups=num_group),
        #     LeakyReLU(leaky_relu_alpha),
        #     Conv2D(out_channel * 4, kernel_size=1, use_bias=False, padding='same'),
        # ])
        self.resnext_conv_block = Sequential([
            Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False),
            LeakyReLU(leaky_relu_alpha),
            GroupConv2D(out_channel, kernel_size=3, use_bias=False, padding='same', groups=groups),
            LeakyReLU(leaky_relu_alpha),
            Conv2D(out_channel, kernel_size=1, use_bias=False, padding='same'),
        ])
        self.out_leaky_relu = LeakyReLU(leaky_relu_alpha)

    def call(self, x):
        out = self.resnext_conv_block(x)
        out += x
        out = self.out_leaky_relu(out)

        return out


class Conv2DSubPixel(Model):
    """Sub-pixel convolution layer.
    See https://arxiv.org/abs/1609.05158
    scale 是倍数
    通道数 // scale ** 2
    通道数需要被整除
    """
    def __init__(self, scale, trainable=False, **kwargs):
        self.scale = scale
        super().__init__(trainable=trainable, **kwargs)

    def call(self, t):
        r = self.scale
        shape = t.shape.as_list()
        new_shape = self.compute_output_shape(shape)
        H, W = shape[1:3]
        C = new_shape[-1]
        t = tf.reshape(t, [-1, H, W, r, r, C])
        # Here we are different from Equation 4 from the paper. That equation
        # is equivalent to switching 3 and 4 in `perm`. But I feel my
        # implementation is more natural.
        t = tf.transpose(t, perm=[0, 1, 3, 2, 4, 5])  # S, H, r, H, r, C
        t = tf.reshape(t, [-1, H * r, W * r, C])
        return t

    def compute_output_shape(self, input_shape):
        r = self.scale
        H, W, rrC = np.array(input_shape[1:])
        assert rrC % (r ** 2) == 0, "通道数 // scale ** 2,且应当被整除"
        return input_shape[0], H * r, W * r, rrC // (r ** 2)


class Up(Model):

    def __init__(self, out_channel, groups=32, scale=2, leaky_relu_alpha=0.2):
        super(Up, self).__init__()
        self.up = Sequential([
            Conv2D(out_channel * 4, kernel_size=1, padding='same', use_bias=False),
            LeakyReLU(leaky_relu_alpha),
            ResNeXtBlock(out_channel * 4, groups, leaky_relu_alpha),
            Conv2DSubPixel(scale)
        ])
        self.conv = DoubleConvBlock(out_channel, leaky_relu_alpha=leaky_relu_alpha)
        # self.conv1 = Conv2D(out_channel * 4, kernel_size=1, padding='same', use_bias=False)
        # self.leaky_relu = LeakyReLU(leaky_relu_alpha)
        # self.resnext_block = ResNeXtBlock(out_channel * 4, groups, leaky_relu_alpha)
        # self.conv2d_sub_pixel = Conv2DSubPixel(scale)

    def call(self, x, local_x):

        # x = self.conv1(x)
        # x = self.leaky_relu(x)
        # x = self.resnext_block(x)
        # x = self.conv2d_sub_pixel(x)

        x = self.up(x)
        x = tf.concat((x, local_x), axis=-1)
        x = self.conv(x)
        return x


class ChangedUNet(Model):

    def __init__(self, leaky_relu_alpha=0.2):
        super(ChangedUNet, self).__init__()
        self.local_conv = DoubleConvBlock(64)

        self.down_cnn_block = DownCNNBlock(512)

        self.down1 = Down(128)
        self.down2 = Down(256)
        self.down3 = Down(512)

        self.up1 = Up(256)
        self.up2 = Up(128)
        self.up3 = Up(64)
        self.out_conv = Conv2D(3, kernel_size=1, padding='same', use_bias=False)
        self.leaky_relu = LeakyReLU(leaky_relu_alpha)

    def call(self, x):

        local_x = self.down_cnn_block(x)
        x = self.local_conv(x)

        x1 = self.down1(x)
        print(x1.shape)
        x2 = self.down2(x1)
        print(x2.shape)
        x3 = self.down3(x2)
        print(x3.shape)
        x3 = tf.concat([x3, local_x], axis=-1)
        print(x3.shape)
        x4 = self.up1(x3, x2)
        print(x4.shape)
        x5 = self.up2(x4, x1)
        print(x5.shape)
        x6 = self.up3(x5, x)
        print(x6.shape)
        x = self.out_conv(x6)
        out = self.leaky_relu(x)

        return out
