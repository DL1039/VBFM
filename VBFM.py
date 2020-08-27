# For disabling the warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import scipy.io
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from datetime import datetime
import os



#Load NN Data
mat=scipy.io.loadmat(r'/home/dl2020/Python/BostonHousing/Data2.mat')
print(mat.keys())

X_train=mat["X_train"]
y_train=mat["y_train"]
X_test=mat["X_test"]
y_test=mat["y_test"]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

initializer = tf.keras.initializers.he_uniform()

# define base model
def baseline_model():

	# create model
	model = Sequential()

	#model.add(Dense(100, input_dim=7, kernel_initializer=initializer, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
	model.add(Dense(100, input_dim=7, kernel_initializer=initializer))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	# model.add(Dense(50, kernel_initializer= initializer, activation=LeakyReLU(alpha=0.1)))
	model.add(Dense(50, kernel_initializer= initializer))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	# model.add(Dense(20, kernel_initializer= initializer, activation=LeakyReLU(alpha=0.1)))
	model.add(Dense(20, kernel_initializer= initializer))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	# model.add(Dense(8, kernel_initializer= initializer, activation=LeakyReLU(alpha=0.1)))
	model.add(Dense(8, kernel_initializer= initializer))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	model.add(Dense(1, kernel_initializer=initializer))

	model.compile(loss='mean_squared_error',metrics=['mse'], optimizer='adam')

	model.summary()
	return model


# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=2, batch_size=64,validation_split = 0.2, verbose=1)
kfold = KFold(n_splits=5)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Results", results)

estimator.fit(X_train, y_train)

loss=estimator.model.evaluate(X_train,y_train)
print('Train loss:',loss)

loss =estimator.model.evaluate(X_test,y_test)
print('Test loss:',loss)

#create a directory with sytem datetime and save the model
mydir = os.path.join(os.getcwd(), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)
dst = os.path.join(os.getcwd(),mydir)
estimator.model.save(dst+'/VBFM2.h5')

predicted_y_train=estimator.model.predict(X_train)
plt.title('Train Data')
plt.plot(X_train[:,0:1], y_train, 'ro', X_train[:,0:1],predicted_y_train,'bs',markersize=1)
#plt.plot(X_train[:,0:1],predicted_y_train,'bs',markersize=1)
#plt.show()
plt.savefig(os.path.join(dst,'Train.png'))

predicted_y_test=estimator.model.predict(X_test)
plt.title('Test Data')
#plt.plot(X_test[:,0:1], y_test, 'ro', X_test[:,0:1],predicted_y_test,'bs',markersize=1)
#plt.show()
plt.savefig(os.path.join(dst,'Test.png'))