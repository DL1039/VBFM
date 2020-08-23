# For disabling the warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from numpy import poly1d
from numpy import polyfit


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

#flattening the data (convert to 1D vector)
x=X_train[:,0].flatten()
y=y_train.flatten()
x_t=X_test[:,0].flatten()
y_t=y_test.flatten()

#visualisation of the results
mymodel = np.poly1d(np.polyfit(x,y,3))
plt.plot(x, y, 'o', x_t,mymodel(x_t), '-')
plt.show()

# Evaluation of the model using mse and mae
mse=mean_squared_error(y_t,mymodel(x_t)) 
print('MSE is:',mse)

mae=mean_absolute_error(y_t,mymodel(x_t))
print('MAE is:',mae)
