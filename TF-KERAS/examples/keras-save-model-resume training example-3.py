import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import os
import glob


"""
Custom model definition
"""
class CustomModel(keras.Model):
    def __init__(self, hidden_units):
        super(CustomModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = []
        for i,u in enumerate(hidden_units):
            if i == len(hidden_units) - 1:
                self.dense_layers.append(keras.layers.Dense(u, activation="sigmoid"))
            else:
                self.dense_layers.append(keras.layers.Dense(u, activation="relu"))
        #self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


"""
Usage Example
"""

# 0: starts training; otherwise, loads a pre-trained model.
initial_epoch = 0

epochs = 20 # number of epochs
log_dir = "./logs_example_3"

"""
Dataset and model definition
"""
N_examples = 5000
N_features = 10
X = tf.random.normal([N_examples,N_features], mean = 0.0, stddev=1.0, dtype=tf.float16)
y = tf.random.uniform([N_examples,1], minval=0, maxval=1)
y = tf.round(y)


# model definition
model_obj = CustomModel([100, 1])

input = tf.keras.Input(shape=(10))
model = tf.keras.Model(input, model_obj(input), name="CustomModel")

print(model.summary())



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
                                       custom_objects={'CustomModel': CustomModel})

    print("AFTER loading the trained model:")
    print("Is the optimizer None?", ("yes" if (model.optimizer is None) else "no"))
    print("Learning rate = ", K.eval(model.optimizer.lr))
    print("The optimizer's config after loading:\n", model.optimizer.get_config())

    print("RESUME TRAINING FROM EPOCH %d ..." % initial_epoch)
else:
    initial_epoch = 0

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, batch_size=32, epochs=epochs,
          callbacks=callbacks_list,
          initial_epoch=initial_epoch)