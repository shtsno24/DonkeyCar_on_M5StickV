"""
This model is based on "KerasCategorical" from Donkey Car Project.
Please Refer to https://github.com/autorope/donkeycar/blob/dev/donkeycar/parts/keras.py

"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Softmax, ReLU


def Categorical(input_shape=(120, 160, 3), drop=0.2, l4_stride=1):
    """
    :param img_in:          input layer of network
    :param drop:            dropout rate
    :param l4_stride:       4-th layer stride, default 1, in Categorical, l4_stride=2
    """
    inputs = Input(shape=input_shape, name='img_in')
    x = conv2d_relu(inputs, 24, 5, 2, 1)
    x = Dropout(drop)(x)
    x = conv2d_relu(x, 32, 5, 2, 2)
    x = Dropout(drop)(x)
    x = conv2d_relu(x, 64, 5, 2, 3)
    x = Dropout(drop)(x)
    x = conv2d_relu(x, 64, 3, l4_stride, 4)
    x = Dropout(drop)(x)
    x = conv2d_relu(x, 64, 3, 1, 5)
    x = Dropout(drop)(x)

    x = Flatten(name='flattened')(x)

    x = Dense(100, name='dense_1')(x)
    x = ReLU()(x)
    x = Dropout(drop)(x)

    x = Dense(50, name='dense_2')(x)
    x = ReLU()(x)
    x = Dropout(drop)(x)

    outputs = []
    _x = Dense(15, name='throttle')(x)
    _x = Softmax()(_x)
    outputs.append(_x)
    _x = Dense(20, name='steer')(x)
    _x = Softmax()(_x)
    outputs.append(_x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def conv2d_relu(x, filters, kernel, strides, layer_num):
    """
    Helper function to create a standard valid-padded convolutional layer
    with square kernel and strides and unified naming convention
    :param filters:     channel dimension of the layer
    :param kernel:      creates (kernel, kernel) kernel matrix dimension
    :param strides:     creates (strides, strides) stride
    :param layer_num:   used in labelling the layer
    """
    x = Conv2D(filters=filters,
               kernel_size=(kernel, kernel),
               strides=(strides, strides),
               name='conv2d_' + str(layer_num))(x)
    x = ReLU()(x)

    return x
