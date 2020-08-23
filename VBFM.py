# For disabling the warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import scipy.io
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot


#Load NN Data
mat=scipy.io.loadmat(r'/home/dl2020/Python/BostonHousing/Data.mat')
print(mat.keys())

X_train=mat["X_train"]
y_train=mat["y_train"]
X_test=mat["X_test"]
y_test=mat["y_test"]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	#model.add(Dense(4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error',metrics=['mse'], optimizer='adam')

	# model = Sequential()
	# # The Input Layer :
	# model.add(Dense(128, kernel_initializer='normal',input_dim = 7, activation='relu'))
	# # The Hidden Layers :
	# model.add(Dense(256, kernel_initializer='normal',activation='relu'))
	# model.add(Dense(256, kernel_initializer='normal',activation='relu'))
	# model.add(Dense(256, kernel_initializer='normal',activation='relu'))
	# # The Output Layer :
	# model.add(Dense(1, kernel_initializer='normal',activation='linear'))
	# # Compile the network :
	# model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

	model.summary()
	return model


# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=8,validation_split = 0.2, verbose=1)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
#print("Baseline: %.2f (%.2f) MSE %.2f" % (results.mean(), results.std(), results.var() ))
print("Results", results)

model = baseline_model()
model.save('VBFM.h5')
