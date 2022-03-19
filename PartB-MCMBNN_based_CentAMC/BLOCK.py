import os
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Dropout, concatenate, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM, Lambda, Concatenate, Activation, Flatten, Multiply, Add, Subtract, CuDNNGRU
from keras import backend as K
import tensorflow as tf


def _group_conv(x, filters, kernel, stride, groups):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]

    # number of input channels per group
    nb_ig = in_channels // groups
    # number of output channels per group
    nb_og = filters // groups

    gc_list = []
    # Determine whether the number of filters is divisible by the number of groups
    assert filters % groups == 0

    for i in range(groups):
        if channel_axis == -1:
            x_group = Lambda(lambda z: z[:, :, i * nb_ig: (i + 1) * nb_ig])(x)
        else:
            x_group = Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :])(x)
        gc_list.append(Conv1D(filters=nb_og, kernel_size=kernel, strides=stride,
                              padding='same', use_bias=False)(x_group))

    return Concatenate(axis=channel_axis)(gc_list)

def _group_conv2d(x, filters, kernel, stride, groups):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]

    # number of input channels per group
    nb_ig = in_channels // groups
    # number of output channels per group
    nb_og = filters // groups

    gc_list = []
    # Determine whether the number of filters is divisible by the number of groups
    assert filters % groups == 0

    for i in range(groups):
        if channel_axis == -1:
            x_group = Lambda(lambda z: z[:, :, :,i * nb_ig: (i + 1) * nb_ig])(x)
        else:
            x_group = Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :])(x)
        gc_list.append(Conv2D(filters=nb_og, kernel_size=kernel, strides=stride,
                              padding='same', use_bias=False)(x_group))

    return Concatenate(axis=channel_axis)(gc_list)

def cal1(x):
    y = tf.keras.backend.cos(x)
    return y

def cal2(x):
    y = tf.keras.backend.sin(x)
    return y
def BLOCK(weights=None,
           input_shape1=[2, 128],
           input_shape2=[128, 1],
           classes=11,
           **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5

    input1 = Input(input_shape1 + [1], name='I/Qchannel')
    input2 = Input(input_shape2, name='Ichannel')
    input3 = Input(input_shape2, name='Qchannel')
    x1f = Flatten()(input1)
    x1f = Dense(1, name='fc2')(x1f)
    x1f = Activation('linear')(x1f)

    cos1= Lambda(cal1)(x1f)
    sin1 = Lambda(cal2)(x1f)
    x11f = Multiply()([input2, cos1])
    x12f = Multiply()([input3, sin1])
    x21f = Multiply()([input3, cos1])
    x22f = Multiply()([input2, sin1])
    y1 = Add()([x11f,x12f])
    y2 = Subtract()([x21f,x22f])
    y1 = Reshape(target_shape=(128, 1), name='reshape1')(y1)
    y2 = Reshape(target_shape=(128, 1), name='reshape2')(y2)
    x11f = concatenate([y1, y2])
    x3f = Reshape(target_shape=((128, 2, 1)), name='reshape3')(x11f)

    xm0 = Conv2D(20, (2, 8), padding='same',activation="relu", kernel_initializer='glorot_normal', data_format='channels_last')(input1)
    # xm0 = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding='valid', data_format='channels_last')(xm0)
    xm1 = Conv2D(20, (8, 2), padding='same',activation="relu", kernel_initializer='glorot_normal', data_format='channels_last')(input1)
    # xm1 = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding='valid', data_format='channels_last')(xm1)
    xm2 = Conv2D(10, (1, 1), padding='same',activation="relu", kernel_initializer='glorot_normal', data_format='channels_last')(input1)
    # xm2 = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding='valid', data_format='channels_last')(xm2)
    xm = concatenate([xm0, xm1], axis=3)
    x1 = concatenate([xm, xm2], axis=3)
    x1 = Activation('relu')(x1)
    x1 = Dropout(dr)(x1)

    xc0 = Conv1D(20, 2, padding='causal',activation="relu", kernel_initializer='glorot_normal', data_format='channels_last')(input2)
    xc1 = Conv1D(20, 4, padding='causal',activation="relu", kernel_initializer='glorot_normal', data_format='channels_last')(input2)
    xc2 = Conv1D(10, 8, padding='causal',activation="relu", kernel_initializer='glorot_normal', data_format='channels_last')(input2)
    xc = concatenate([xc0, xc1], axis=2)
    x2 = concatenate([xc, xc2], axis=2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(dr)(x2)
    x2_reshape = Reshape([-1, 128, 50])(x2)

    xd0 = Conv1D(20, 2, padding='causal', kernel_initializer='glorot_normal', data_format='channels_last')(input3)
    xd1 = Conv1D(20, 4, padding='causal', kernel_initializer='glorot_normal', data_format='channels_last')(input3)
    xd2 = Conv1D(10, 8, padding='causal', kernel_initializer='glorot_normal', data_format='channels_last')(input3)
    xd = concatenate([xd0, xd1], axis=2)
    xd = concatenate([xd, xd2], axis=2)
    xd = Activation('relu')(xd)
    xd = Dropout(dr)(xd)
    x3_reshape = Reshape([-1, 128, 50], name="reshap2")(xd)

    x = concatenate([x2_reshape, x3_reshape], axis=1, name='Concatenate1')
    xm0 = Conv2D(20, (1, 8), padding='same',activation="relu", kernel_initializer='glorot_normal', data_format='channels_last')(x)
    xm1 = Conv2D(20, (8, 1), padding='same',activation="relu", kernel_initializer='glorot_normal', data_format='channels_last')(x)
    xm2 = Conv2D(10, (1, 1), padding='same',activation="relu", kernel_initializer='glorot_normal', data_format='channels_last')(x)
    xm = concatenate([xm0, xm1], axis=3)
    x = concatenate([xm, xm2], axis=3)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)
    x = concatenate([x1, x], name="Concatenate2")
    x = _group_conv2d(x, filters=50, kernel=(3, 3), stride=(1, 1), groups=2)
    x = Activation('relu')(x)
    x3f = Reshape([2, 128,1])(x3f)
    x = Add()([x3f, x])
    x = Reshape(target_shape=((256, 50)))(x)
    x = CuDNNGRU(units=128)(x)
    x = Dropout(dr)(x)
    x = Dense(classes, activation="softmax", name="Softmax")(x)
    model = Model(inputs=[input1, input2, input3], outputs=x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model
import keras
from keras.optimizers import adam

if __name__ == '__main__':
    # for the RaioML2016.10a dataset
    model = BLOCK(classes=11)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
