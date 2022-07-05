# MIRNet model, copied from https://github.com/keras-team/keras-io/blob/master/examples/vision/mirnet.py

import tensorflow as tf
from tensorflow import keras
from keras import layers

def selective_kernel_feature_fusion(
    multi_scale_feature_1, multi_scale_feature_2, multi_scale_feature_3
):
    channels = list(multi_scale_feature_1.shape)[-1]
    combined_feature = layers.Add()([multi_scale_feature_1, multi_scale_feature_2, multi_scale_feature_3])
    gap = layers.GlobalAveragePooling2D()(combined_feature)
    channel_wise_statistics = tf.reshape(gap, shape=(-1, 1, 1, channels))
    compact_feature_representation = layers.Conv2D(filters=max(1, channels // 8), kernel_size=(1, 1), activation="relu")(channel_wise_statistics)
    feature_descriptor_1 = layers.Conv2D(channels, kernel_size=(1, 1), activation="softmax")(compact_feature_representation)
    feature_descriptor_2 = layers.Conv2D(channels, kernel_size=(1, 1), activation="softmax")(compact_feature_representation)
    feature_descriptor_3 = layers.Conv2D(channels, kernel_size=(1, 1), activation="softmax")(compact_feature_representation)
    feature_1 = multi_scale_feature_1 * feature_descriptor_1
    feature_2 = multi_scale_feature_2 * feature_descriptor_2
    feature_3 = multi_scale_feature_3 * feature_descriptor_3
    aggregated_feature = layers.Add()([feature_1, feature_2, feature_3])

    return aggregated_feature

def spatial_attention_block(input_tensor):
    average_pooling = tf.reduce_max(input_tensor, axis=-1)
    average_pooling = tf.expand_dims(average_pooling, axis=-1)
    max_pooling = tf.reduce_mean(input_tensor, axis=-1)
    max_pooling = tf.expand_dims(max_pooling, axis=-1)
    concatenated = layers.Concatenate(axis=-1)([average_pooling, max_pooling])
    feature_map = layers.Conv2D(1, kernel_size=(1, 1))(concatenated)
    feature_map = tf.nn.sigmoid(feature_map)

    return input_tensor * feature_map

def channel_attention_block(input_tensor):
    channels = list(input_tensor.shape)[-1]
    average_pooling = layers.GlobalAveragePooling2D()(input_tensor)
    feature_descriptor = tf.reshape(average_pooling, shape=(-1, 1, 1, channels))
    #feature_activations = layers.Conv2D(filters=channels // 8, kernel_size=(1, 1), activation="relu")(feature_descriptor)
    feature_activations = layers.Conv2D(filters=max(1, channels // 8), kernel_size=(1, 1), activation="relu")(feature_descriptor)
    feature_activations = layers.Conv2D(filters=channels, kernel_size=(1, 1), activation="sigmoid")(feature_activations)

    return input_tensor * feature_activations

def dual_attention_unit_block(input_tensor):
    channels = list(input_tensor.shape)[-1]
    feature_map = layers.Conv2D(channels, kernel_size=(3, 3), padding="same", activation="relu")(input_tensor)
    feature_map = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(feature_map)
    channel_attention = channel_attention_block(feature_map)
    spatial_attention = spatial_attention_block(feature_map)
    concatenation = layers.Concatenate(axis=-1)([channel_attention, spatial_attention])
    concatenation = layers.Conv2D(channels, kernel_size=(1, 1))(concatenation)

    return layers.Add()([input_tensor, concatenation])

def down_sampling_module(input_tensor):
    channels = list(input_tensor.shape)[-1]
    main_branch = layers.Conv2D(channels, kernel_size=(1, 1), activation="relu")(input_tensor)
    main_branch = layers.Conv2D(channels, kernel_size=(3, 3), padding="same", activation="relu")(main_branch)
    main_branch = layers.MaxPooling2D()(main_branch)
    main_branch = layers.Conv2D(channels * 2, kernel_size=(1, 1))(main_branch)
    skip_branch = layers.MaxPooling2D()(input_tensor)
    skip_branch = layers.Conv2D(channels * 2, kernel_size=(1, 1))(skip_branch)

    return layers.Add()([skip_branch, main_branch])

def up_sampling_module(input_tensor):
    channels = list(input_tensor.shape)[-1]
    main_branch = layers.Conv2D(channels, kernel_size=(1, 1), activation="relu")(input_tensor)
    main_branch = layers.Conv2D(channels, kernel_size=(3, 3), padding="same", activation="relu")(main_branch)
    main_branch = layers.UpSampling2D()(main_branch)
    main_branch = layers.Conv2D(channels // 2, kernel_size=(1, 1))(main_branch)
    skip_branch = layers.UpSampling2D()(input_tensor)
    skip_branch = layers.Conv2D(channels // 2, kernel_size=(1, 1))(skip_branch)

    return layers.Add()([skip_branch, main_branch])

def multi_scale_residual_block(input_tensor, channels):
    # features
    level1 = input_tensor
    level2 = down_sampling_module(input_tensor)
    level3 = down_sampling_module(level2)
    
    # DAU
    level1_dau = dual_attention_unit_block(level1)
    level2_dau = dual_attention_unit_block(level2)
    level3_dau = dual_attention_unit_block(level3)
    
    # SKFF
    level1_skff = selective_kernel_feature_fusion(level1_dau, up_sampling_module(level2_dau), up_sampling_module(up_sampling_module(level3_dau)))
    level2_skff = selective_kernel_feature_fusion(down_sampling_module(level1_dau), level2_dau, up_sampling_module(level3_dau))
    level3_skff = selective_kernel_feature_fusion(down_sampling_module(down_sampling_module(level1_dau)), down_sampling_module(level2_dau), level3_dau)
    
    # DAU 2
    level1_dau_2 = dual_attention_unit_block(level1_skff)
    level2_dau_2 = up_sampling_module((dual_attention_unit_block(level2_skff)))
    level3_dau_2 = up_sampling_module(up_sampling_module(dual_attention_unit_block(level3_skff)))
    
    # SKFF 2
    skff_ = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)
    conv = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(skff_)
    return layers.Add()([input_tensor, conv])

def recursive_residual_group(input_tensor, num_mrb, channels):
    conv1 = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(input_tensor)
    for _ in range(num_mrb):
        conv1 = multi_scale_residual_block(conv1, channels)

    conv2 = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(conv1)
    return layers.Add()([conv2, input_tensor])

def mirnet_model(num_rrg=1, num_mrb=1, channels=16, input_shape=[None, None, 1]):
    input_tensor = keras.Input(shape=input_shape)
    x1 = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(input_tensor)
    for _ in range(num_rrg):
        x1 = recursive_residual_group(x1, num_mrb, channels)

    conv = layers.Conv2D(1, kernel_size=(3, 3), padding="same")(x1)
    output_tensor = layers.Add()([input_tensor, conv])
    return keras.Model(input_tensor, output_tensor)
