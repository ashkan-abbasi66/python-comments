"""

BINARY CLASSIFICATION OF MNIST IS TOO SIMPLE FOR A CNN

this is not a good example.

FIND A GOOD EXAMPLE OF A BINARY CNN CLASSIFIER AND TEST IT HERE

"""


from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD

# DOWNLOAD DATASET

# a workaround to avoid multiple downloads of the dataset.
from joblib import Memory
memory = Memory(r'E:\THESIS\ANOTES\ml_in_practice\Classification\tmp')
fetch_openml_cached = memory.cache(fetch_openml)

# load data from https://www.openml.org/d/554
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html
X, y = fetch_openml_cached('mnist_784', version=1,
                           return_X_y=True,
                           data_home = r'E:\THESIS\ANOTES\ml_in_practice\DATASETS\mnist')

# PREPARE TRAIN AND TEST SETS

print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

X_data = X.values.astype('float32')/255.0
data_shape = (X.shape[0], 28, 28, 1)
X_data = X_data.reshape(data_shape)
y_data = (y == 7)

# downsample data
X_data = X_data[:, 0:28:2, 0:28:2, :]
data_shape = X_data.shape

le = LabelEncoder()
le.fit(y_data)
y_data = le.transform(y_data)


def baseline_cnn():
    model = Sequential()
    # model.add(layers.Conv2D(32, (3, 3), kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    # model.add(layers.Dense(100, kernel_initializer='he_uniform'))
    # model.add(layers.Dense(2))
    # model.add(layers.Activation('relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    print(model)

    return model

input_shape = data_shape[1:]
estimator = KerasClassifier(build_fn=baseline_cnn, epochs=5, batch_size=100, verbose=1) # bs = 32
kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(estimator, X_data, y_data, cv=kfold)
print("\nINFO:")
print(results)
print("Baseline CNN: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))