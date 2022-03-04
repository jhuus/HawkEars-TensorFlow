'''
EfficientNetV2 Model as defined in: Mingxing Tan, Quoc V. Le. (2021). arXiv preprint arXiv:2104.00298.
EfficientNetV2: Smaller Models and Faster Training.

This implementation copied from https://github.com/leondgarse/keras_cv_attention_models,
and then modified. Original license is:

MIT License

Copyright (c) 2021 leondgarse

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")

TF_BATCH_NORM_EPSILON = 0.001
TORCH_BATCH_NORM_EPSILON = 1e-5

def hard_swish(inputs):
    """ `out = xx * relu6(xx + 3) / 6`, arxiv: https://arxiv.org/abs/1905.02244 """
    return inputs * tf.nn.relu6(inputs + 3) / 6

def mish(inputs):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function.
    Paper: [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
    Copied from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/activations/mish.py
    """
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

def phish(inputs):
    """Phish is defined as f(x) = xTanH(GELU(x)) with no discontinuities in the f(x) derivative.
    Paper: https://www.techrxiv.org/articles/preprint/Phish_A_Novel_Hyper-Optimizable_Activation_Function/17283824
    """
    return inputs * tf.math.tanh(tf.nn.gelu(inputs))

def activation_by_name(inputs, activation="relu", name=None):
    """ Typical Activation layer added hard_swish and prelu. """
    layer_name = name and activation and name + activation
    if activation == "hard_swish":
        return keras.layers.Activation(activation=hard_swish, name=layer_name)(inputs)
    elif activation == "mish":
        return keras.layers.Activation(activation=mish, name=layer_name)(inputs)
    elif activation == "phish":
        return keras.layers.Activation(activation=phish, name=layer_name)(inputs)
    elif activation.lower() == "prelu":
        shared_axes = list(range(1, len(inputs.shape)))
        shared_axes.pop(-1 if K.image_data_format() == "channels_last" else 0)
        return keras.layers.PReLU(shared_axes=shared_axes, alpha_initializer=tf.initializers.Constant(0.25), name=layer_name)(inputs)
    elif activation.lower().startswith("gelu/app"):
        # gelu/approximate
        return tf.nn.gelu(inputs, approximate=True, name=layer_name)
    elif activation:
        return keras.layers.Activation(activation=activation, name=layer_name)(inputs)
    else:
        return inputs

def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, epsilon=BATCH_NORM_EPSILON, act_first=False, name=None):
    """ Performs a batch normalization followed by an activation. """
    bn_axis = -1 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    if act_first and activation:
        inputs = activation_by_name(inputs, activation=activation, name=name)
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=epsilon,
        gamma_initializer=gamma_initializer,
        name=name and name + "bn",
    )(inputs)
    if not act_first and activation:
        nn = activation_by_name(nn, activation=activation, name=name)
    return nn

def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", use_bias=False, groups=1, use_torch_padding=True, name=None, **kwargs):
    """ Typical Conv2D with `use_bias` default as `False` and fixed padding """
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if use_torch_padding and padding.upper() == "SAME" and pad != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
        padding = "VALID"

    groups = max(1, groups)
    if groups == filters:
        return keras.layers.DepthwiseConv2D(
            kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "conv", **kwargs
        )(inputs)
    else:
        return keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            groups=groups,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name and name + "conv",
            **kwargs,
        )(inputs)

def drop_block(inputs, drop_rate=0, name=None):
    """ Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382 """
    if drop_rate > 0:
        noise_shape = [None] + [1] * (len(inputs.shape) - 1)  # [None, 1, 1, 1]
        return keras.layers.Dropout(drop_rate, noise_shape=noise_shape, name=name and name + "drop")(inputs)
    else:
        return inputs

def make_divisible(vv, divisor=4, min_value=None):
    """ Copied from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(vv + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * vv:
        new_v += divisor
    return new_v
    
def output_block(inputs, num_features=0, activation="relu", num_classes=1000, drop_rate=0, classifier_activation="softmax", is_torch_mode=True):
    nn = inputs
    if num_features > 0:  # efficientnet like
        bn_eps = BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON
        nn = conv2d_no_bias(nn, num_features, 1, strides=1, use_torch_padding=is_torch_mode, name="features_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="features_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)
    return nn

def se_module(inputs, se_ratio=0.25, divisor=8, activation="relu", use_bias=True, name=None):
    """ Squeeze-and-Excitation block, arxiv: https://arxiv.org/pdf/1709.01507.pdf """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = int(filters * se_ratio)
    reduction = make_divisible(filters * se_ratio, divisor)
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "1_conv")(se)
    se = activation_by_name(se, activation=activation, name=name)
    se = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "2_conv")(se)
    se = activation_by_name(se, activation="sigmoid", name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, se])

def MBConv(inputs, output_channel, stride, expand_ratio, shortcut, kernel_size=3, drop_rate=0, use_se=0, is_fused=False, is_torch_mode=False, activation="swish", name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_channel = inputs.shape[channel_axis]
    bn_eps = TORCH_BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON

    if is_fused and expand_ratio != 1:
        nn = conv2d_no_bias(inputs, input_channel * expand_ratio, 3, stride, padding="same", use_torch_padding=is_torch_mode, name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name + "sortcut_")
    elif expand_ratio != 1:
        nn = conv2d_no_bias(inputs, input_channel * expand_ratio, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name + "sortcut_")
    else:
        nn = inputs

    if not is_fused:
        if is_torch_mode and kernel_size // 2 > 0:
            nn = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(nn)
            padding = "VALID"
        else:
            padding = "SAME"
        nn = keras.layers.DepthwiseConv2D(kernel_size, padding=padding, strides=stride, use_bias=False, name=name + "MB_dw_")(nn)
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name + "MB_dw_")

    if use_se:
        nn = se_module(nn, se_ratio=1 / (4 * expand_ratio), divisor=1, activation=activation, name=name + "se_")

    # pw-linear
    if is_fused and expand_ratio == 1:
        nn = conv2d_no_bias(nn, output_channel, 3, strides=stride, padding="same", use_torch_padding=is_torch_mode, name=name + "fu_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name + "fu_")
    else:
        nn = conv2d_no_bias(nn, output_channel, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name=name + "MB_pw_")
        nn = batchnorm_with_activation(nn, activation=None, epsilon=bn_eps, name=name + "MB_pw_")

    if shortcut:
        nn = drop_block(nn, drop_rate, name=name + "drop")
        return keras.layers.Add(name=name + "output")([inputs, nn])
    else:
        return keras.layers.Activation("linear", name=name + "output")(nn)  # Identity, Just need a name here

def EfficientNetV2(
    model_type,
    input_shape=(None, None, 3),
    num_classes=1000,
    activation="swish",
    dropout=0.2,
    first_strides=2,
    is_torch_mode=False,
    drop_connect_rate=0,
    classifier_activation="softmax",
    model_name="EfficientNet",
    kwargs=None,
):
    blocks_config = BLOCK_CONFIGS.get(model_type.lower())
    expands = blocks_config["expands"]
    out_channels = blocks_config["out_channels"]
    depths = blocks_config["depths"]
    strides = blocks_config["strides"]
    use_ses = blocks_config["use_ses"]
    first_conv_filter = blocks_config.get("first_conv_filter", out_channels[0])
    output_conv_filter = blocks_config.get("output_conv_filter", 1280)
    kernel_sizes = blocks_config.get("kernel_sizes", [3] * len(depths))

    inputs = keras.layers.Input(shape=input_shape)
    bn_eps = TORCH_BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON

    nn = inputs
    out_channel = make_divisible(first_conv_filter, 8)
    nn = conv2d_no_bias(nn, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="stem_")

    pre_out = out_channel
    global_block_id = 0
    total_blocks = sum(depths)
    for id, (expand, out_channel, depth, stride, se, kernel_size) in enumerate(zip(expands, out_channels, depths, strides, use_ses, kernel_sizes)):
        out = make_divisible(out_channel, 8)
        is_fused = True if se == 0 else False
        for block_id in range(depth):
            stride = stride if block_id == 0 else 1
            shortcut = True if out == pre_out and stride == 1 else False
            name = "stack_{}_block{}_".format(id, block_id)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = MBConv(nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, activation, name=name)
            pre_out = out
            global_block_id += 1

    output_conv_filter = make_divisible(output_conv_filter, 8)
    nn = conv2d_no_bias(nn, output_conv_filter, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name="post_")
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="post_")
    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)

    model = keras.models.Model(inputs=inputs, outputs=nn, name=model_name)
    return model

# Configurations a## were added for HawkEars; 
# Using squeeze-and-excitation blocks (use_ses=1) makes the model smaller but slower
BLOCK_CONFIGS = {
    "a0": {  # custom 3-layer (fastest option)
        "first_conv_filter": 32,
        "expands": [1, 2, 4],
        "out_channels": [8, 16, 32],
        "depths": [1, 2, 2],
        "strides": [1, 2, 2],
        "use_ses": [0, 0, 1],
    },
    "a1": {  # custom 4-layer (~374K trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 2, 4, 4],
        "out_channels": [16, 32, 48, 64],
        "depths": [1, 2, 2, 3],
        "strides": [1, 2, 2, 2],
        "use_ses": [0, 0, 0, 1],
    },
    "a2": {  # custom 4-layer (~1.04M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 6],
        "out_channels": [16, 32, 64, 128],
        "depths": [1, 2, 2, 3],
        "strides": [1, 2, 2, 2],
        "use_ses": [0, 0, 0, 1],
    },
    "a3": {  # custom 5-layer (~1.5M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 112],
        "depths": [1, 2, 2, 3, 5],
        "strides": [1, 2, 2, 2, 1],
        "use_ses": [0, 0, 0, 1, 1],
    },
    "a4": {  # custom 6-layer (~2M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 96, 112],
        "depths": [1, 2, 2, 3, 5, 5],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "a5": {  # custom 6-layer (~2.2M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 96, 112],
        "depths": [1, 2, 2, 3, 5, 6],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "a6": {  # custom 6-layer (~2.4M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 96, 112],
        "depths": [1, 2, 2, 3, 5, 7],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "a7": {  # custom 6-layer (~2.6M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 96, 112],
        "depths": [1, 2, 2, 3, 5, 8],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "a8": {  # custom 6-layer (~2.7M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 112, 128],
        "depths": [1, 2, 2, 3, 5, 5],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "a9": {  # custom 6-layer (~3.0M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 4, 6],
        "out_channels": [16, 32, 48, 96, 112, 128],
        "depths": [1, 2, 2, 3, 5, 7],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "b0": {  # width 1.0, depth 1.0 (~5.8M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depths": [1, 2, 2, 3, 5, 8],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "b1": {  # width 1.0, depth 1.1 (~6.8M trainable parameters)
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depths": [2, 3, 3, 4, 6, 9],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "b2": {  # width 1.1, depth 1.2 (~8.7M trainable parameters)
        "first_conv_filter": 32,
        "output_conv_filter": 1408,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 56, 104, 120, 208],
        "depths": [2, 3, 3, 4, 6, 10],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "b3": {  # width 1.2, depth 1.4
        "first_conv_filter": 40,
        "output_conv_filter": 1536,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 40, 56, 112, 136, 232],
        "depths": [2, 3, 3, 5, 7, 12],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "t": {  # width 1.4 * 0.8, depth 1.8 * 0.9, from timm
        "first_conv_filter": 24,
        "output_conv_filter": 1024,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 40, 48, 104, 128, 208],
        "depths": [2, 4, 4, 6, 9, 14],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "s": {  # width 1.4, depth 1.8
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 48, 64, 128, 160, 256],
        "depths": [2, 4, 4, 6, 9, 15],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "early": {  # S model discribed in paper early version https://arxiv.org/pdf/2104.00298v2.pdf
        "first_conv_filter": 24,
        "output_conv_filter": 1792,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 48, 64, 128, 160, 272],
        "depths": [2, 4, 4, 6, 9, 15],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
    },
    "m": {  # width 1.6, depth 2.2
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [24, 48, 80, 160, 176, 304, 512],
        "depths": [3, 5, 5, 7, 14, 18, 5],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
    },
    "l": {  # width 2.0, depth 3.1
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 224, 384, 640],
        "depths": [4, 7, 7, 10, 19, 25, 7],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
    },
    "xl": {
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 256, 512, 640],
        "depths": [4, 8, 8, 16, 24, 32, 8],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
    },
}
