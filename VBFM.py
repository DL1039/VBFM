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
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys


#Load NN Data
mat=scipy.io.loadmat(r'/home/dl2020/Python/ML/projects/VBFM/Data2.mat')
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
	model.add(Dense(20, input_dim=7, kernel_initializer=initializer))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	# model.add(Dense(50, kernel_initializer= initializer, activation=LeakyReLU(alpha=0.1)))
	model.add(Dense(15, kernel_initializer= initializer))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	# model.add(Dense(20, kernel_initializer= initializer, activation=LeakyReLU(alpha=0.1)))
	model.add(Dense(8, kernel_initializer= initializer))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	
	model.add(Dense(1, kernel_initializer=initializer))

	model.compile(loss='mean_squared_error', optimizer='adam')

	model.summary()
	return model

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)

batch_size=64
epochs=20

# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=batch_size,validation_split = 0.2, verbose=1)
kfold = KFold(n_splits=5)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Results", results)

History=estimator.fit(X_train, y_train,callbacks=[es_callback])


print (History.history.keys())
# summarize history for loss
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
#plt.show()

#create a directory with sytem datetime and save the model
mydir = os.path.join(os.getcwd(), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)
dst = os.path.join(os.getcwd(),mydir)
estimator.model.save(dst+'/VBFM2.h5')
plt.savefig(os.path.join(dst,'Loss.png'))

#Evaluation of the model for the training data
predicted_y_train=estimator.model.predict(X_train)
loss=estimator.model.evaluate(X_train,y_train)
r_2=r2_score(y_train, predicted_y_train)
print("\nTrain Loss: %.4f \nTrain R_squared: %.2f" %(loss,r_2))

#log the evaluation metrics for training data
f = open(dst+"/eval_metrics.txt", "w")
f.write("\nTrain Loss: %.4f \nTrain R_squared: %.2f" %(loss,r_2))

plt.clf()
plt.title('Train Data')
plt.plot(X_train[:,0:1], y_train, 'ro', X_train[:,0:1],predicted_y_train,'bs',markersize=1)
#plt.show()
plt.savefig(os.path.join(dst,'Train.png'))


#Evaluation of the model for the test data
predicted_y_test=estimator.model.predict(X_test)
loss=estimator.model.evaluate(X_test,y_test)
r_2=r2_score(y_test, predicted_y_test)
print("\nTest Loss: %.4f \nTest R_squared: %.2f" %(loss,r_2))

#log the evaluation metrics for test data
f.write("\nTest Loss: %.4f \nTest R_squared: %.2f\n" %(loss,r_2))

plt.clf()
plt.title('Test Data')
plt.plot(X_test[:,0:1], y_test, 'ro', X_test[:,0:1],predicted_y_test,'bs',markersize=1)
#plt.show()
plt.savefig(os.path.join(dst,'Test.png'))

#log the model summary
f.write("\nBatch size: %d \nNumber of epochs: %d\n\n" %(batch_size,epochs))
orig_stdout = sys.stdout
sys.stdout = f
print(estimator.model.summary())
sys.stdout = orig_stdout
f.close()