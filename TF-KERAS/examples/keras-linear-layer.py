"""

GOAL: create a linear layer with very basic KERAS / tensorflow API

"""
import tensorflow as tf

w = tf.Variable(
    initial_value=tf.random.normal([2,1], 0, 1, tf.float32, seed=1),
    trainable=True)

n=10
f = tf.constant(tf.random.normal([n,2], 0, 1, tf.float32, seed=1))

z = tf.matmul(f,w)

print(z)

N = 100
X = tf.random.normal([N, n, 2], 0, 1, tf.float32, seed=1) # NWC
Y = tf.maximum(X[:,:,0], X[:,:,1])

print(Y)

tf.keras.backend.variable()


model = tf.keras.Model(input = Input)
tf.train

