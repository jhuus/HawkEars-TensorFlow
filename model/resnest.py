# ResNeSt model from the paper "ResNeSt: Split-Attention Networks", Zhang et al [2020].
# This code was copied from https://github.com/QiaoranC/tf_ResNeSt_RegNet_model 
# and then substantially modified.

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

from tensorflow.keras import models
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
    multiply,
    Permute,
    Reshape,
)

# We could do grouped convolution using Conv2D(groups=xx, ...), but as of Tensorflow 2.5 that requires
# a GPU, even for running analysis. There is an apparent fix in 2.7 (see 
# https://github.com/tensorflow/tensorflow/commit/7b8db6083b34520688dbc71f341f7aeaf156bf17),
# but in my testing that still failed when no GPU was available.
class GroupedConv2D(object):
    # filters: Integer, the dimensionality of the output space.
    # kernel_size: An integer or a list. If it is a single integer, then it is
    #    same as the original Conv2D. If it is a list, then we split the channels
    #    and perform different kernel for each group.
    # **kwargs: other parameters passed to the original conv2d layer.
    def __init__(self, filters, kernel_size, **kwargs):
        self._groups = len(kernel_size)
        self._convs = []
        splits = self._split_channels(filters, self._groups)
        for i in range(self._groups):
            self._convs.append(Conv2D(filters=splits[i], kernel_size=kernel_size[i], **kwargs))

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def __call__(self, inputs):
        if len(self._convs) == 1:
            return self._convs[0](inputs)

        filters = inputs.shape[-1]
        splits = self._split_channels(filters, len(self._convs))
        x_splits = tf.split(inputs, splits, -1)
        x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
        x = tf.concat(x_outputs, -1)
        return x

class ResNest:
    # num_stages is from 1 to 4;
    # blocks_set gives the number of blocks per stage (so array with length >= num_stages);
    # setting deep_stem = True adds about 43K trainable parameters (see model differences below)
    # avg_down = True means do downsampling with AveragePooling instead of in the Conv2D layer;
    # avd_first determines whether the extra AveragePooling layer is inserted before or after the split attention block
    def __init__(self, num_classes=3, input_shape=(80, 384, 1), active='mish', dropout_rate=0.5,
                 num_stages=1, blocks_set=[1], radix=2, groups=1, deep_stem=True, stem_width=32,
                 bottleneck_width=64, kernel_size=3, avg_down=True, avd_first=True, seed=None):
                 
        self.channel_axis = -1
        self.input_shape = input_shape
        self.active = active
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        if active == 'mish':
            self.active = tfa.activations.mish
        else:
            self.active = active # e.g. 'relu'

        self.num_stages = num_stages
        self.blocks_set = blocks_set
        self.radix = radix
        self.groups = groups
        self.bottleneck_width = bottleneck_width

        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.kernel_size = kernel_size
        self.avg_down = avg_down
        self.avd_first = avd_first
        self.seed = seed
        self.initializer = tf.keras.initializers.HeNormal(seed=self.seed)

    # make the initial set of layers
    def _make_stem(self, input_tensor):
        x = input_tensor
        if self.deep_stem:
            x = Conv2D(self.stem_width, kernel_size=3, strides=2, padding='same', kernel_initializer=self.initializer,
                       use_bias=False, data_format='channels_last')(x)

            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)
            x = Conv2D(self.stem_width, kernel_size=3, strides=1, padding='same',
                       kernel_initializer=self.initializer, use_bias=False, data_format='channels_last')(x)
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)
            x = Conv2D(self.stem_width * 2, kernel_size=3, strides=1, padding='same', kernel_initializer=self.initializer,
                       use_bias=False, data_format='channels_last')(x)
        else:
            x = Conv2D(self.stem_width, kernel_size=7, strides=2, padding='same', kernel_initializer=self.initializer,
                       use_bias=False, data_format='channels_last')(x)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
            
        return x

    def _rsoftmax(self, input_tensor, filters):
        x = input_tensor
        batch = x.shape[0]
        if self.radix > 1:
            x = tf.reshape(x, [-1, self.groups, self.radix, filters // self.groups])
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.keras.activations.softmax(x, axis=1)
            x = tf.reshape(x, [-1, 1, 1, self.radix * filters])
        elif self.num_classes > 2:
            x = Activation('sigmoid')(x)
        else:
            x = Activation('softmax')(x)
            
        return x

    # split-attention Conv2D
    def _SplAtConv2d(self, input_tensor, filters=64, kernel_size=3, stride=1):
        x = input_tensor
        in_channels = input_tensor.shape[-1]

        '''
        # Tensorflow grouped convolution, which requires a GPU  
        x = Conv2D(filters=filters * self.radix, kernel_size=kernel_size, 
                   groups=self.groups * self.radix,
                   padding='same', kernel_initializer=self.initializer, use_bias=False,
                   data_format='channels_last')(x)
        '''
        
        # use this version in case no GPU is available
        x = GroupedConv2D(filters=filters * self.radix, kernel_size=[kernel_size for i in range(self.groups * self.radix)], 
                          padding='same', kernel_initializer=self.initializer, use_bias=False,
                          data_format='channels_last')(x)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        batch, rchannel = x.shape[0], x.shape[-1]
        if self.radix > 1:
            splits = tf.split(x, self.radix, axis=-1)
            gap = sum(splits)
        else:
            gap = x

        gap = GlobalAveragePooling2D(data_format='channels_last')(gap)
        gap = tf.reshape(gap, [-1, 1, 1, filters])

        reduction_factor = 4
        inter_channels = max(in_channels * self.radix // reduction_factor, 32)

        x = Conv2D(inter_channels, kernel_size=1)(gap)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(filters * self.radix, kernel_size=1)(x)

        atten = self._rsoftmax(x, filters)

        if self.radix > 1:
            logits = tf.split(atten, self.radix, axis=-1)
            out = sum([a * b for a, b in zip(splits, logits)])
        else:
            out = atten * x
        return out

    # make a block within a stage
    def _make_block(self, input_tensor, first_block=True, filters=64, stride=2, is_first=False):
        x = input_tensor
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        short_cut = x
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters:
            if self.avg_down:
                short_cut = AveragePooling2D(pool_size=stride, strides=stride, padding='same', data_format='channels_last')(short_cut)
                short_cut = Conv2D(filters, kernel_size=1, strides=1, padding='same', kernel_initializer=self.initializer,
                                   use_bias=False, data_format='channels_last')(short_cut)
            else:
                short_cut = Conv2D(filters, kernel_size=1, strides=stride, padding='same', kernel_initializer=self.initializer,
                                   use_bias=False, data_format='channels_last')(short_cut)

        group_width = int(filters * (self.bottleneck_width / 64.0)) * self.groups
        avd = stride > 1 or is_first

        if avd:
            avd_layer = AveragePooling2D(pool_size=3, strides=stride, padding='same', data_format='channels_last')
            stride = 1

        if avd and self.avd_first:
            x = avd_layer(x)

        if self.radix >= 1:
            x = self._SplAtConv2d(x, filters=group_width, kernel_size=self.kernel_size, stride=stride)
        else:
            x = Conv2D(filters, kernel_size=3, strides=stride, padding='same', kernel_initializer=self.initializer,
                       use_bias=False, data_format='channels_last')(x)

        if avd and not self.avd_first:
            x = avd_layer(x)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer=self.initializer,
                   use_bias=False, data_format='channels_last')(x)
            
        m2 = Add()([x, short_cut])
        return m2

    # make a group of blocks
    def _make_stage(self, input_tensor, blocks=4, filters=64, stride=2, is_first=True):
        x = input_tensor
        x = self._make_block(x, first_block=True, filters=filters, stride=stride, is_first=is_first)
        for i in range(1, blocks):
            x = self._make_block(x, first_block=False, filters=filters, stride=1)
        return x

    def build_model(self):
        input_sig = Input(shape=self.input_shape)
        x = self._make_stem(input_sig)
        x = MaxPool2D(pool_size=3, strides=2, padding='same', data_format='channels_last')(x)
        x = self._make_stage(x, blocks=self.blocks_set[0], filters=64, stride=1, is_first=False)

        b1_b3_filters = [64,128,256,512]
        for i in range(self.num_stages - 1):
            idx = i+1
            x = self._make_stage(x, blocks=self.blocks_set[idx], filters=b1_b3_filters[idx], stride=2)

        x = GlobalAveragePooling2D(name='avg_pool')(x) 

        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate, noise_shape=None, seed=self.seed)(x)

        fc_out = Dense(self.num_classes, activation='softmax', kernel_initializer=self.initializer, use_bias=False, dtype='float32', name='fc_NObias')(x)
        model = models.Model(inputs=input_sig, outputs=fc_out)

        return model
