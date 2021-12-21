"""

RESULT:
Mean of accuracies: 84.45% (11.48%)

"""

import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# LOAD DATA

dataframe = read_csv("./sonar_data/archive/sonar.csv")
dataset = dataframe.values

X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

# PREPARE DATA

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)




def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=60, activation='relu'))
	model.add(Dense(2, activation='softmax'))

	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	return model


# EVALUATE THE MODEL

clf = Pipeline([
	('standardize', StandardScaler()),
	('mlp', KerasClassifier(build_fn=create_baseline, epochs=100,batch_size=5, verbose=0))
				])
kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = []
for i, (train_indices, val_indices) in enumerate(kfold.split(X, encoded_Y)):

	print("\nINFO: Training the model ...")
	X_train = X[train_indices, :]
	X_val = X[val_indices, :]

	y_train = encoded_Y[train_indices]
	y_val = encoded_Y[val_indices]

	y_train_onehot = to_categorical(y_train, dtype="uint8")
	y_val_onehot = to_categorical(y_val, dtype="uint8")

	clf.fit(X_train, y_train_onehot)

	y_val_pred = clf.predict(X_val)
	results.append(accuracy_score(y_val, y_val_pred))

print("Mean of accuracies: %.2f%% (%.2f%%)" % (np.mean(results)*100, np.std(results)*100))
