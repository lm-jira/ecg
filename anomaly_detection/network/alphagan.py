import tensorflow as tf
# Parameter for train/test/inference
learning_rate = 0.001
batch_size = 100
latent_dim = 46

# Infile parameter
init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.2)
dis_inter_layer_dim = 500


def leakyReLu(x, alpha=0.2):
        return tf.nn.leaky_relu(x, alpha=alpha)


def conv_layer(x, layer_num, padding="SAME", filters=32):
    layer_name = "conv1d_"+layer_num
    return tf.layers.conv1d(x,
                            filters=filters,
                            kernel_size=30,
                            padding=padding,
                            name=layer_name,
                            kernel_initializer=init_kernel)


def _encoder_block(x, number, is_training):
    temp_x = x
    x = conv_layer(x, layer_num="{}_1".format(number))
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, training=is_training)

    x = conv_layer(x, layer_num="{}_2".format(number))

    x = x + temp_x
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, training=is_training)

    x = tf.expand_dims(x, 2)
    x = tf.nn.max_pool(x,
                       ksize=[1, 5, 1, 1],
                       strides=[1, 2, 1, 1],
                       padding='VALID')
    shape = x.shape
    x = tf.reshape(x, [-1, shape[1], shape[3]])

    return x


def encoder(x_inp, is_training, reuse=False):
    with tf.variable_scope("encoder", reuse=reuse):
        x = x_inp  # x is an array with shape [184, 1]
        x = conv_layer(x, layer_num="0")
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training)

        x = conv_layer(x, layer_num="1")
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training)

        for i in range(5):
            x = _encoder_block(x, i+1, is_training)

        x = tf.reshape(x, [-1, x.shape[1]*x.shape[2]])
        x = tf.layers.dense(x,
                            latent_dim,
                            kernel_initializer=init_kernel,
                            name='fc')

    return x


def _generator_block(z, number, is_training, filters=64, activation="relu"):
    temp_z = z
    z = conv_layer(z,
                   layer_num="{}_1".format(number))
    z = tf.nn.relu(z)
    z = tf.layers.batch_normalization(z, training=is_training)
    z = z + temp_z

    z = conv_layer(z,
                   layer_num="{}_2".format(number),
                   filters=filters)

    shape = z.shape
    z = tf.reshape(z, [-1, shape[1]*2, shape[2]//2])

    if activation == "sigmoid":
        z = tf.sigmoid(z, name='sigmoid')
    elif activation == "relu":
        z = tf.nn.relu(z)
        z = tf.layers.batch_normalization(z, training=is_training)
    else:
        print("Undefined activation function in generator block")
        exit()

    return z


def generator(z_inp, is_training, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        z = z_inp  # z is an array with shape [batch_size, 23]

        z = tf.layers.dense(z, 100, kernel_initializer=init_kernel)
        z = tf.nn.relu(z)
        z = tf.layers.batch_normalization(z, training=is_training)

        z = tf.layers.dense(z, 200, kernel_initializer=init_kernel)
        z = tf.nn.relu(z)
        z = tf.layers.batch_normalization(z, training=is_training)

        z = tf.layers.dense(z, 23, kernel_initializer=init_kernel)
        z = tf.nn.relu(z)
        z = tf.layers.batch_normalization(z, training=is_training)

        z = tf.expand_dims(z, 2)

        z = conv_layer(z, layer_num="0")
        z = tf.nn.relu(z)
        z = tf.layers.batch_normalization(z, training=is_training)

        for i in range(2):
            z = _generator_block(z, i+1, is_training)

        z = _generator_block(z, 3, is_training, filters=2, activation="sigmoid")

    return z


def discriminator(x_inp, is_training, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        with tf.variable_scope("x"):
            x = x_inp
            x = conv_layer(x, layer_num="0")

            x = conv_layer(x, layer_num="1", filters=64)
            x = tf.nn.relu(x)
            x = tf.expand_dims(x, 2)
            x = tf.nn.max_pool(x,
                               ksize=[1, 5, 1, 1],
                               strides=[1, 2, 1, 1],
                               padding='VALID')
            shape = x.shape
            x = tf.reshape(x, [-1, shape[1], shape[3]])

            x = conv_layer(x, layer_num="2", filters=128)
            x = tf.nn.relu(x)
            x = tf.expand_dims(x, 2)
            x = tf.nn.max_pool(x,
                               ksize=[1, 5, 1, 1],
                               strides=[1, 2, 1, 1],
                               padding='VALID')
            shape = x.shape
            x = tf.reshape(x, [-1, shape[1], shape[3]])

            x = conv_layer(x, layer_num="3", filters=256)
            x = tf.nn.relu(x)
            x = tf.expand_dims(x, 2)
            x = tf.nn.max_pool(x,
                               ksize=[1, 5, 1, 1],
                               strides=[1, 2, 1, 1],
                               padding='VALID')
            shape = x.shape
            x = tf.reshape(x, [-1, shape[1], shape[3]])

        x = tf.reshape(x, [-1, x.shape[1]*x.shape[2]])

        y = tf.layers.dense(x,
                            dis_inter_layer_dim,
                            kernel_initializer=init_kernel,
                            name='fc1')
        y = tf.nn.relu(y)
        y = tf.layers.dropout(y, rate=0.5, name='dropout',
                              training=is_training)

        y = tf.layers.dense(y,
                            1,
                            kernel_initializer=init_kernel,
                            name='fc2')
        y = tf.sigmoid(y)
    return y


def code_discriminator(z_inp, reuse=False):
    """ Discriminator architecture in tensorflow
    Discriminates between pairs (E(x), x) and (z, G(z))
    Args:
        x_inp (tensor): input data for the discriminator.
        reuse (bool): sharing variables or not
    Returns:
        logits (tensor): last activation layer of the discriminator (shape 1)
    """
    with tf.variable_scope('code_discriminator', reuse=reuse):
        z = z_inp
        h = 250
        # latent_dim = 128

        z = tf.layers.dense(z, h, kernel_initializer=init_kernel)
        z = leakyReLu(z)

        z = tf.layers.dense(z, h, kernel_initializer=init_kernel)
        z = leakyReLu(z)

        z = tf.layers.dense(z, 1, kernel_initializer=init_kernel)
        z = tf.sigmoid(z)

    return z
