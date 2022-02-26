"""
Goal:
    How to save a model with custom loss function

    Custom loss function is defined by a class inherited from Loss

Usage:
    First, run this script with "initial_epoch = 0".
    Then, based on the saved checkpoints, set the "initial_epoch" to an
    appropriate value to load the associated trained model.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K

"""
Custom loss function definition
"""
from tensorflow.keras.losses import Loss
class MyBinCrossEntropy(Loss):
    def __init__(self, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='myBinCrossEntropy'): # myBinCrossEntropy
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits

    # compute loss
    def call(self, y_true, y_pred):
        bin_crossentropy = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        return bin_crossentropy

    def get_config(self):
        config = super(MyBinCrossEntropy, self).get_config()
        return config


"""
Usage Example
"""

import os
import glob

# 0: starts training; otherwise, loads a pre-trained model.
initial_epoch = 0

epochs = 20 # number of epochs
log_dir = "./logs_example_2"

"""
Dataset and model definition
"""
N_examples = 5000
N_features = 10
X = tf.random.normal([N_examples,N_features], mean = 0.0, stddev=1.0, dtype=tf.float16)
y = tf.random.uniform([N_examples,1], minval=0, maxval=1)
y = tf.round(y)

# model definition
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(N_features,)))
model.add(tf.keras.layers.Dense(units=100, activation="relu"))
model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

model.summary()

loss = MyBinCrossEntropy()



"""
callbacks
"""
metric_name = "accuracy"
check_filename = "epoch={epoch:02d}#%s={%s:.4f}" % (
    metric_name, metric_name
)
check_filepath = os.path.join(log_dir, check_filename)
# checkpoint callback
check_cb = tf.keras.callbacks.ModelCheckpoint(check_filepath,
                                           monitor="%s"%metric_name, # Usually a metric on the validation set is used here
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode="max",
                                           save_freq='epoch')

def step_decay_at(epoch):
    decay_rate = 0.5
    decay_at = 10 # every 10 epochs, reduce the LR
    lr0 = 0.01 # initial learning rate
    lr = lr0 * (decay_rate ** (epoch // decay_at))
    return lr
learning_rate_cb = LearningRateScheduler(step_decay_at)
callbacks_list = [check_cb, learning_rate_cb]


"""
training
"""

if initial_epoch>0:
    file_list = glob.glob(os.path.join(log_dir, "epoch=%02d*" % initial_epoch))
    assert len(file_list) > 0, "There is NO checkpoint file. Check the \"file_list\"."
    assert len(file_list) == 1, "More than one checkpoints are selected for loading"

    print("BEFORE loading the trained model:")
    print("Is the optimizer None?", ("Yes" if (model.optimizer is None) else "No"))

    model = tf.keras.models.load_model(file_list[0],
                                       custom_objects={'MyBinCrossEntropy': MyBinCrossEntropy})

    print("AFTER loading the trained model:")
    print("Is the optimizer None?", ("yes" if (model.optimizer is None) else "no"))
    print("Learning rate = ", K.eval(model.optimizer.lr))
    print("The optimizer's config after loading:\n", model.optimizer.get_config())

    print("RESUME TRAINING FROM EPOCH %d ..." % initial_epoch)
else:
    initial_epoch = 0

model.compile(optimizer='adam', loss=loss, metrics=["accuracy"])
model.fit(X, y, batch_size=32, epochs=epochs,
          callbacks=callbacks_list,
          initial_epoch=initial_epoch)


