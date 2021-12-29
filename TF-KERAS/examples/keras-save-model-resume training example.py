import tensorflow as tf
import os
import glob

description = "example_of_resume_training"
log_dir = './logs/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_dir = (log_dir + '{}').format(description)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

X = tf.random.normal([5000,10], mean = 0.0, stddev=1.0, dtype=tf.float16)
y = tf.random.uniform([5000,1], minval=0, maxval=1)
y = tf.round(y)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(10,)))
model.add(tf.keras.layers.Dense(units=100, activation="relu"))
model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

model.summary()

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=["accuracy"])

metric_name = "accuracy"
weights_name = "epoch={epoch:02d}#%s={%s:.4f}.h5" % (
    metric_name, metric_name
)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(log_dir, weights_name),
                                monitor="%s"%metric_name, # Usually a metric on the validation set is used here
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False,
                                mode="max",
                                save_freq='epoch',save_format='tf')

initial_epoch = 6
if initial_epoch>0:
    file_list = glob.glob(os.path.join(log_dir, "epoch=%02d*.h5" % initial_epoch))
    assert len(file_list) > 0, "There is no .h5 file. Check the \"file_list\"."
    assert len(file_list) == 1, "More than one checkpoint is selected for loading"
    model = tf.keras.models.load_model(file_list[0])
else:
    initial_epoch = 0

model.fit(X, y, batch_size=32, epochs=10,
          callbacks=[checkpoint_cb],
          initial_epoch=initial_epoch)

