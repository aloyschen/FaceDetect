import tensorflow as tf

def make_var(name, shape, trainable = True):
    """
    Introduction
    ------------
        创建变量
    Parameters
    ----------
        name: 变量名字
        shape: 变量形状
        trainable: 是否为训练变量
    Returns
    -------
        tensorflow变量
    """
    return tf.get_variable(name = name, shape = shape, trainable = trainable)

def conv(name, input_layer, kernel_size, channels_output, stride, padding = 'SAME', biased = True, relu = True):
    """
    Introduction
    ------------
        卷积层
    Parameters
    ----------
        name: 卷积层名字
        input_layer: 卷积层输入
        kernel_size: 卷积层kernel大小
        channels_output: 卷积层输出通道数
        stride: 卷积层步长
        padding: 卷积层padding策略
        biased: 是否加入偏置项
        relu: 是否加入relu激活函数
    Returns
    -------
        卷积层计算结果
    """
    channels_input = int(input_layer.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        kernel = make_var('weights', shape = [kernel_size[1], kernel_size[0], channels_input, channels_output])
        output = tf.nn.conv2d(input_layer, kernel, strides = [1, stride[1], stride[0], 1], padding = padding)
        if biased:
            bias = make_var('biases', shape = [channels_output])
            output = tf.nn.bias_add(output, bias)
        if relu:
            output = tf.nn.relu(output, name = scope.name)
    return output


def prelu(name, input):
    """
    Introduction
    ------------
    Parameters
    ----------
        input: 输入变量
        name: 层的名字
    Returns
    -------
        激活函数计算结果
    """
    with tf.variable_scope(name):
        channels_input = int(input.get_shape()[-1])
        alpha = make_var('alpha', shape = [channels_input])
        output = tf.nn.relu(input) + tf.multiply(alpha, -tf.nn.relu(-input))
    return output


def max_pool(name, input, kernel_size, stride, padding = 'SAME'):
    """
    Introduction
    ------------
        池化层
    Parameters
    ----------
        name: 命名
        input: 输入变量
        kernel_size: 池化层kernel大小
        stride: 池化层步长
        padding: 池化层padding方式
    Returns
    -------
        池化层计算结果
    """
    output = tf.nn.max_pool(input, ksize = [1, kernel_size[1], kernel_size[0], 1], strides = [1, stride[1], stride[0], 1], padding = padding, name = name)
    return output


def vectorize_input(input_layer):
    """
    Introduction
    ------------
        对全连接层输入做reshape操作
    Parameters
    ----------
        input_layer: 输入层变量
    """
    input_shape = input_layer.get_shape()

    if input_shape.ndims == 4:
        dim = 1
        for x in input_shape[1:].as_list():
            dim *= int(x)
        vectorized_input = tf.reshape(input_layer, [-1, dim])
    else:
        vectorized_input, dim = (input_layer, input_shape[-1].value)

    return vectorized_input, dim


def fc(name, input, channels_output, relu = True):
    """
    Introduction
    ------------
        全连接层
    Parameters
    ----------
        name: 命名
        input: 输入变量
        channels_output: 输出通道数量
        relu: 是否使用relu激活函数
    Returns
    -------
        全连接层计算结果
    """
    with tf.variable_scope(name):
        vectorized_input, dimension = vectorize_input(input)
        weights = make_var('weights', shape = [dimension, channels_output])
        bias = make_var('biases', shape = [channels_output])
        operation = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        fc = operation(vectorized_input, weights, bias, name = name)
    return fc

