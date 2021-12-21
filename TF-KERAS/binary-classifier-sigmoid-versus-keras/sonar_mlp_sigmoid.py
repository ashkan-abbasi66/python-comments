"""

FOUND AT
Binary Classification Tutorial with the Keras Deep Learning Library
https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

DATASET:
https://datahub.io/machine-learning/sonar#resource-sonar_zip

RESULT:
84.10% (7.81%)

"""

from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
	model.add(Dense(1, activation='sigmoid'))

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
results = cross_val_score(clf, X, encoded_Y, cv=kfold)

print("Mean of accuracies: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
