# Simple Example:
# input/output shape: N*H*W*C
# filter shape: h*w*(# of FMs in the previous layer)*(# of FMs); FMs: feature maps
#
# with tf.variable_scope("generator"):
#     W1=tf.Variable(tf.truncated_normal([9, 9, 3, 64], stddev=0.01), name="W1")
#     conv1=tf.nn.conv2d(input, W1, strides=[1, 1, 1, 1], padding='SAME')
#     b1=tf.Variable(tf.constant(0.01, shape=64),name="b1")
#     conv1=conv1+b1
#     conv1=tf.nn.relu(conv1)
#
#     -------------- Residual Block 1 --------------
#     W2=tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01), name="W2")
#     conv2=tf.nn.conv2d(conv1, W2, strides=[1, 1, 1, 1], padding='SAME')
#     b2=tf.Variable(tf.constant(0.01, shape=64),name="b2")
#     conv2=conv2+b2
#     conv2=_instance_norm(conv2) # batch normalization
#     conv2=tf.nn.relu(conv2)
#
#     W3=tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01), name="W3")
#     conv3=tf.nn.conv2d(conv2, W3, strides=[1, 1, 1, 1], padding='SAME')
#     b3=tf.Variable(tf.constant(0.01, shape=64),name="b3")
#     conv3=conv3+b3
#     conv3=_instance_norm(conv3) # batch normalization
#     conv3=tf.nn.relu(conv3)
#
#     conv3=conv3+conv1 # skip connection
#     -----------------------------------------------
#     ....(see the rest of the code)


# Summary of the structure:
# i -> C1: C(9*9),R -> C2: C(3*3),BN,R -> C3: (C,BN,R)+C1 -> C4: C,BN,R -> C5: (C,BN,R)+C3
#   -> C6: C,BN,R -> C7: (C,BN,R)+C5 -> C8: C,BN,R -> C9: (C,BN,R)+C7
#   -> C10: C,R   -> C11: C,R -> C12: C(9*9),tanh*0.58+0.5 (enhanced image - 3 channels)

import tensorflow as tf

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def _instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift


def resnet(input_image):

    with tf.variable_scope("generator"):

        W1 = weight_variable([9, 9, 3, 64], name="W1")
        b1 = bias_variable([64], name="b1")
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

        # residual 1

        W2 = weight_variable([3, 3, 64, 64], name="W2")
        b2 = bias_variable([64], name="b2")
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 64, 64], name="W3")
        b3 = bias_variable([64], name="b3")
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1

        # residual 2

        W4 = weight_variable([3, 3, 64, 64], name="W4")
        b4 = bias_variable([64], name="b4")
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))

        W5 = weight_variable([3, 3, 64, 64], name="W5")
        b5 = bias_variable([64], name="b5")
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5)) + c3

        # residual 3

        W6 = weight_variable([3, 3, 64, 64], name="W6")
        b6 = bias_variable([64], name="b6")
        c6 = tf.nn.relu(_instance_norm(conv2d(c5, W6) + b6))

        W7 = weight_variable([3, 3, 64, 64], name="W7")
        b7 = bias_variable([64], name="b7")
        c7 = tf.nn.relu(_instance_norm(conv2d(c6, W7) + b7)) + c5

        # residual 4

        W8 = weight_variable([3, 3, 64, 64], name="W8")
        b8 = bias_variable([64], name="b8")
        c8 = tf.nn.relu(_instance_norm(conv2d(c7, W8) + b8))

        W9 = weight_variable([3, 3, 64, 64], name="W9")
        b9 = bias_variable([64], name="b9")
        c9 = tf.nn.relu(_instance_norm(conv2d(c8, W9) + b9)) + c7

        # Convolutional

        W10 = weight_variable([3, 3, 64, 64], name="W10")
        b10 = bias_variable([64], name="b10")
        c10 = tf.nn.relu(conv2d(c9, W10) + b10)

        W11 = weight_variable([3, 3, 64, 64], name="W11")
        b11 = bias_variable([64], name="b11")
        c11 = tf.nn.relu(conv2d(c10, W11) + b11)

        # Final

        W12 = weight_variable([9, 9, 64, 3], name="W12")
        b12 = bias_variable([3], name="b12")
        enhanced = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return enhanced

# enhanced = models.resnet(phone_image)
# loss_generator = w_content * loss_content + w_texture * loss_texture + w_color * loss_color + w_tv * loss_tv
# generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
# train_step_gen = tf.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)
# saver = tf.train.Saver(var_list=generator_vars, max_to_keep=100)
# sess.run(tf.global_variables_initializer())
# ....
