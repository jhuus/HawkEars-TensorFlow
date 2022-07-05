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
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
LAYER_NORM_EPSILON = 1e-5
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

def add_pre_post_process(model, rescale_mode="tf", input_shape=None, post_process=None):
    input_shape = model.input_shape[1:-1] if input_shape is None else input_shape
    model.preprocess_input = PreprocessInput(input_shape, rescale_mode=rescale_mode)
    model.decode_predictions = imagenet_decode_predictions if post_process is None else post_process
    model.rescale_mode = rescale_mode

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

def depthwise_conv2d_no_bias(inputs, kernel_size, strides=1, padding="VALID", use_bias=False, use_torch_padding=True, name=None, **kwargs):
    """ Typical DepthwiseConv2D with `use_bias` default as `False` and fixed padding """
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

# mc_dropout is Monte Carlo dropout
def drop_block(inputs, drop_rate=0, name=None, mc_dropout=None):
    """ Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382 """
    if drop_rate > 0:
        noise_shape = [None] + [1] * (len(inputs.shape) - 1)  # [None, 1, 1, 1]
        return keras.layers.Dropout(drop_rate, noise_shape=noise_shape, name=name and name + "drop")(inputs, training=mc_dropout)
    else:
        return inputs

def drop_connect_rates_split(num_blocks, start=0.0, end=0.0):
    """ split drop connect rate in range `(start, end)` according to `num_blocks` """
    drop_connect_rates = tf.split(tf.linspace(start, end, sum(num_blocks)), num_blocks)
    return [ii.numpy().tolist() for ii in drop_connect_rates]

def global_context_module(inputs, use_attn=True, ratio=0.25, divisor=1, activation="relu", use_bias=True, name=None):
    """ Global Context Attention Block, arxiv: https://arxiv.org/pdf/1904.11492.pdf """
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

def layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, name=None):
    """ Typical LayerNormalization with epsilon=1e-5 """
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    return keras.layers.LayerNormalization(axis=norm_axis, epsilon=epsilon, name=name and name + "ln")(inputs)

def make_divisible(vv, divisor=4, min_value=None, limit_round_down=0.9):
    """ Copied from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(vv + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < limit_round_down * vv:
        new_v += divisor
    return new_v

def mlp_block(inputs, hidden_dim, output_channel=-1, drop_rate=0, use_conv=False, activation="gelu", name=None):
    output_channel = output_channel if output_channel > 0 else inputs.shape[-1]
    if use_conv:
        nn = keras.layers.Conv2D(hidden_dim, kernel_size=1, use_bias=True, name=name and name + "Conv_0")(inputs)
    else:
        nn = keras.layers.Dense(hidden_dim, name=name and name + "Dense_0")(inputs)
    nn = activation_by_name(nn, activation, name=name)
    nn = keras.layers.Dropout(drop_rate)(nn) if drop_rate > 0 else nn
    if use_conv:
        nn = keras.layers.Conv2D(output_channel, kernel_size=1, use_bias=True, name=name and name + "Conv_1")(nn)
    else:
        nn = keras.layers.Dense(output_channel, name=name and name + "Dense_1")(nn)
    nn = keras.layers.Dropout(drop_rate)(nn) if drop_rate > 0 else nn
    return nn
    
def multi_head_self_attention(
    inputs, num_heads=4, key_dim=0, out_shape=None, out_weight=True, qkv_bias=False, out_bias=False, attn_dropout=0, output_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = float(1.0 / tf.math.sqrt(tf.cast(key_dim, "float32")))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    qk_out = num_heads * key_dim
    vv_dim = out_shape // num_heads

    qkv = keras.layers.Dense(qk_out * 2 + out_shape, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    qkv = tf.reshape(qkv, [-1, qkv.shape[1] * qkv.shape[2], qkv.shape[-1]])
    query, key, value = tf.split(qkv, [qk_out, qk_out, out_shape], axis=-1)
    query = tf.transpose(tf.reshape(query, [-1, query.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  #  [batch, num_heads, hh * ww, key_dim]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, vv_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, vv_dim]

    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale  # [batch, num_heads, hh * ww, hh * ww]
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
    attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores) if attn_dropout > 0 else attention_scores

    # value = [batch, num_heads, hh * ww, vv_dim], attention_output = [batch, num_heads, hh * ww, vv_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * vv_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * vv_dim] * [num_heads * vv_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    attention_output = keras.layers.Dropout(output_dropout, name=name and name + "out_drop")(attention_output) if output_dropout > 0 else attention_output
    return attention_output

# mc_dropout is Monte Carlo dropout
def output_block(inputs, num_features=0, activation="relu", num_classes=1000, drop_rate=0, classifier_activation="softmax", is_torch_mode=True, mc_dropout=None):
    nn = inputs
    if num_features > 0:  # efficientnet like
        bn_eps = BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON
        nn = conv2d_no_bias(nn, num_features, 1, strides=1, use_torch_padding=is_torch_mode, name="features_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="features_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate, name="head_drop")(nn, training=mc_dropout)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)
    return nn

def se_module(inputs, se_ratio=0.25, divisor=8, limit_round_down=0.9, activation="relu", use_bias=True, name=None):
    """ Squeeze-and-Excitation block, arxiv: https://arxiv.org/pdf/1709.01507.pdf """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    # activation could be ("relu", "hard_sigmoid") for mobilenetv3
    hidden_activation, output_activation = activation if isinstance(activation, (list, tuple)) else (activation, "sigmoid")
    filters = inputs.shape[channel_axis]
    reduction = make_divisible(filters * se_ratio, divisor, limit_round_down=limit_round_down)
    # print(f"{filters = }, {se_ratio = }, {divisor = }, {reduction = }")
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "1_conv")(se)
    se = activation_by_name(se, activation=hidden_activation, name=name)
    se = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "2_conv")(se)
    se = activation_by_name(se, activation=output_activation, name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, se])

class ChannelAffine(keras.layers.Layer):
    def __init__(self, use_bias=True, weight_init_value=1, **kwargs):
        super(ChannelAffine, self).__init__(**kwargs)
        self.use_bias, self.weight_init_value = use_bias, weight_init_value
        self.ww_init = keras.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
        self.bb_init = "zeros"
        self.supports_masking = False

    def build(self, input_shape):
        self.ww = self.add_weight(name="weight", shape=(input_shape[-1],), initializer=self.ww_init, trainable=True)
        if self.use_bias:
            self.bb = self.add_weight(name="bias", shape=(input_shape[-1],), initializer=self.bb_init, trainable=True)
        super(ChannelAffine, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * self.ww + self.bb if self.use_bias else inputs * self.ww

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ChannelAffine, self).get_config()
        config.update({"use_bias": self.use_bias, "weight_init_value": self.weight_init_value})
        return config

class MixupToken(keras.layers.Layer):
    def __init__(self, scale=2, beta=1.0, **kwargs):
        super(MixupToken, self).__init__(**kwargs)
        self.scale, self.beta = scale, beta

    def call(self, inputs, training=None, **kwargs):
        height, width = tf.shape(inputs)[1], tf.shape(inputs)[2]
        # tf.print("training:", training)
        def _call_train():
            return tf.stack(self.rand_bbox(height, width))

        def _call_test():
            return tf.cast(tf.stack([0, 0, 0, 0]), "int32")  # No mixup area for test

        return K.in_train_phase(_call_train, _call_test, training=training)

    def sample_beta_distribution(self):
        gamma_1_sample = tf.random.gamma(shape=[], alpha=self.beta)
        gamma_2_sample = tf.random.gamma(shape=[], alpha=self.beta)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    def rand_bbox(self, height, width):
        random_lam = tf.cast(self.sample_beta_distribution(), self.compute_dtype)
        cut_rate = tf.sqrt(1.0 - random_lam)
        s_height, s_width = height // self.scale, width // self.scale

        right_pos = tf.random.uniform(shape=[], minval=0, maxval=s_width, dtype=tf.int32)
        bottom_pos = tf.random.uniform(shape=[], minval=0, maxval=s_height, dtype=tf.int32)
        left_pos = right_pos - tf.cast(tf.cast(s_width, cut_rate.dtype) * cut_rate, "int32") // 2
        top_pos = bottom_pos - tf.cast(tf.cast(s_height, cut_rate.dtype) * cut_rate, "int32") // 2
        left_pos, top_pos = tf.maximum(left_pos, 0), tf.maximum(top_pos, 0)

        return left_pos, top_pos, right_pos, bottom_pos

    def do_mixup_token(self, inputs, bbox):
        left, top, right, bottom = bbox
        # if tf.equal(right, 0) or tf.equal(bottom, 0):
        #     return inputs
        sub_ww = inputs[:, :, left:right]
        mix_sub = tf.concat([sub_ww[:, :top], sub_ww[::-1, top:bottom], sub_ww[:, bottom:]], axis=1)
        output = tf.concat([inputs[:, :, :left], mix_sub, inputs[:, :, right:]], axis=2)
        output.set_shape(inputs.shape)
        return output

    def get_config(self):
        config = super(MixupToken, self).get_config()
        config.update({"scale": self.scale, "beta": self.beta})
        return config

class PreprocessInput:
    """ `rescale_mode` `torch` means `(image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]`, `tf` means `(image - 0.5) / 0.5` """

    def __init__(self, input_shape=(224, 224, 3), rescale_mode="torch"):
        self.rescale_mode = rescale_mode
        self.input_shape = input_shape[1:-1] if len(input_shape) == 4 else input_shape[:2]

    def __call__(self, image, resize_method="bilinear", resize_antialias=False, input_shape=None):
        input_shape = self.input_shape if input_shape is None else input_shape[:2]
        image = tf.convert_to_tensor(image)
        if tf.reduce_max(image) < 2:
            image *= 255
        image = tf.image.resize(image, input_shape, method=resize_method, antialias=resize_antialias)
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)

        if self.rescale_mode == "raw":
            return image
        elif self.rescale_mode == "raw01":
            return image / 255.0
        else:
            return tf.keras.applications.imagenet_utils.preprocess_input(image, mode=self.rescale_mode)

