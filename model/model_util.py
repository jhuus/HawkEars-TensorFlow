'''
Copied from https://github.com/leondgarse/keras_cv_attention_models,
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
TF_BATCH_NORM_EPSILON = 0.001
TORCH_BATCH_NORM_EPSILON = 1e-5
LAYER_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")

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
    """Typical Activation layer added hard_swish and prelu."""
    if activation is None:
        return inputs

    layer_name = name and activation and name + activation
    activation_lower = activation.lower()
    if activation_lower == "hard_swish":
        return keras.layers.Activation(activation=hard_swish, name=layer_name)(inputs)
    elif activation_lower == "mish":
        return keras.layers.Activation(activation=mish, name=layer_name)(inputs)
    elif activation_lower == "phish":
        return keras.layers.Activation(activation=phish, name=layer_name)(inputs)
    elif activation_lower == "prelu":
        shared_axes = list(range(1, len(inputs.shape)))
        shared_axes.pop(-1 if K.image_data_format() == "channels_last" else 0)
        # print(f"{shared_axes = }")
        return keras.layers.PReLU(shared_axes=shared_axes, alpha_initializer=tf.initializers.Constant(0.25), name=layer_name)(inputs)
    elif activation_lower.startswith("gelu/app"):
        # gelu/approximate
        return tf.nn.gelu(inputs, approximate=True, name=layer_name)
    elif activation_lower.startswith("leaky_relu/"):
        # leaky_relu with alpha parameter
        alpha = float(activation_lower.split("/")[-1])
        return keras.layers.LeakyReLU(alpha=alpha, name=layer_name)(inputs)
    else:
        return keras.layers.Activation(activation=activation, name=layer_name)(inputs)

def batchnorm_with_activation(
    inputs, activation=None, zero_gamma=False, epsilon=1e-5, momentum=0.9, act_first=False, use_evo_norm=False, evo_norm_group_size=-1, name=None
):
    """Performs a batch normalization followed by an activation."""
    if use_evo_norm:
        nonlinearity = False if activation is None else True
        num_groups = inputs.shape[-1] // evo_norm_group_size  # Currently using gorup_size as parameter only
        return EvoNormalization(nonlinearity, num_groups=num_groups, zero_gamma=zero_gamma, epsilon=epsilon, momentum=momentum, name=name + "evo_norm")(inputs)

    bn_axis = -1 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    if act_first and activation:
        inputs = activation_by_name(inputs, activation=activation, name=name)
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=momentum,
        epsilon=epsilon,
        gamma_initializer=gamma_initializer,
        name=name and name + "bn",
    )(inputs)
    if not act_first and activation:
        nn = activation_by_name(nn, activation=activation, name=name)
    return nn

def conv2d_no_bias(inputs, filters, kernel_size=1, strides=1, padding="VALID", use_bias=False, groups=1, use_torch_padding=True, name=None, **kwargs):
    """Typical Conv2D with `use_bias` default as `False` and fixed padding"""
    pad = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, (list, tuple)) else (kernel_size // 2, kernel_size // 2)
    if use_torch_padding and padding.upper() == "SAME" and max(pad) != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
        padding = "VALID"

    groups = max(1, groups)
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

def depthwise_conv2d_no_bias(inputs, kernel_size, strides=1, padding="VALID", use_bias=False, use_torch_padding=True, name=None, **kwargs):
    """Typical DepthwiseConv2D with `use_bias` default as `False` and fixed padding"""
    pad = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, (list, tuple)) else (kernel_size // 2, kernel_size // 2)
    if use_torch_padding and padding.upper() == "SAME" and max(pad) != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "dw_pad")(inputs)
        padding = "VALID"
    return keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "dw_conv",
        **kwargs,
    )(inputs)

def drop_block(inputs, drop_rate=0, name=None):
    """Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382"""
    if drop_rate > 0:
        noise_shape = [None] + [1] * (len(inputs.shape) - 1)  # [None, 1, 1, 1]
        return keras.layers.Dropout(drop_rate, noise_shape=noise_shape, name=name and name + "drop")(inputs)
    else:
        return inputs

def drop_connect_rates_split(num_blocks, start=0.0, end=0.0):
    """split drop connect rate in range `(start, end)` according to `num_blocks`"""
    drop_connect_rates = tf.split(tf.linspace(start, end, sum(num_blocks)), num_blocks)
    return [ii.numpy().tolist() for ii in drop_connect_rates]

def global_context_module(inputs, use_attn=True, ratio=0.25, divisor=1, activation="relu", use_bias=True, name=None):
    """Global Context Attention Block, arxiv: https://arxiv.org/pdf/1904.11492.pdf"""
    height, width, filters = inputs.shape[1], inputs.shape[2], inputs.shape[-1]

    # activation could be ("relu", "hard_sigmoid")
    hidden_activation, output_activation = activation if isinstance(activation, (list, tuple)) else (activation, "sigmoid")
    reduction = make_divisible(filters * ratio, divisor, limit_round_down=0.0)

    if use_attn:
        attn = keras.layers.Conv2D(1, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "attn_conv")(inputs)
        attn = tf.reshape(attn, [-1, 1, 1, height * width])  # [batch, height, width, 1] -> [batch, 1, 1, height * width]
        attn = tf.nn.softmax(attn, axis=-1)
        context = tf.reshape(inputs, [-1, 1, height * width, filters])
        context = attn @ context  # [batch, 1, 1, filters]
    else:
        context = tf.reduce_mean(inputs, [1, 2], keepdims=True)

    mlp = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, name=name and name + "mlp_1_conv")(context)
    mlp = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=name and name + "ln")(mlp)
    mlp = activation_by_name(mlp, activation=hidden_activation, name=name)
    mlp = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, name=name and name + "mlp_2_conv")(mlp)
    mlp = activation_by_name(mlp, activation=output_activation, name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, mlp])

def imagenet_decode_predictions(preds, top=5):
    preds = preds.numpy() if isinstance(preds, tf.Tensor) else preds
    return tf.keras.applications.imagenet_utils.decode_predictions(preds, top=top)

def layer_norm(inputs, zero_gamma=False, epsilon=LAYER_NORM_EPSILON, name=None):
    """Typical LayerNormalization with epsilon=1e-5"""
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    return keras.layers.LayerNormalization(axis=norm_axis, epsilon=epsilon, gamma_initializer=gamma_initializer, name=name and name + "ln")(inputs)

def make_divisible(vv, divisor=4, min_value=None, limit_round_down=0.9):
    """Copied from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(vv + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < limit_round_down * vv:
        new_v += divisor
    return new_v

def output_block(inputs, filters=0, activation="relu", num_classes=1000, drop_rate=0, classifier_activation="softmax", is_torch_mode=True, act_first=False):
    nn = inputs
    if filters > 0:  # efficientnet like
        bn_eps = BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON
        nn = conv2d_no_bias(nn, filters, 1, strides=1, use_bias=act_first, use_torch_padding=is_torch_mode, name="features_")  # Also use_bias for act_first
        nn = batchnorm_with_activation(nn, activation=activation, act_first=act_first, epsilon=bn_eps, name="features_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)
    return nn

def se_module(inputs, se_ratio=0.25, divisor=8, limit_round_down=0.9, activation="relu", use_bias=True, use_conv=True, name=None):
    """Squeeze-and-Excitation block, arxiv: https://arxiv.org/pdf/1709.01507.pdf"""
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    # activation could be ("relu", "hard_sigmoid") for mobilenetv3
    hidden_activation, output_activation = activation if isinstance(activation, (list, tuple)) else (activation, "sigmoid")
    filters = inputs.shape[channel_axis]
    reduction = make_divisible(filters * se_ratio, divisor, limit_round_down=limit_round_down)
    # print(f"{filters = }, {se_ratio = }, {divisor = }, {reduction = }")
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    if use_conv:
        se = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "1_conv")(se)
    else:
        se = keras.layers.Dense(reduction, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "1_dense")(se)
    se = activation_by_name(se, activation=hidden_activation, name=name)
    if use_conv:
        se = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "2_conv")(se)
    else:
        se = keras.layers.Dense(filters, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "2_dense")(se)
    se = activation_by_name(se, activation=output_activation, name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, se])
