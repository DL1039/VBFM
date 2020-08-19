import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import to_categorical

#Load NN Data
mat=scipy.io.loadmat(r'/home/dl2020/Python/NeuralNetwork/NNData.mat')
print(mat.keys())


NNData=mat["NNData"]
print(NNData.shape)

X=NNData[:,0:5]
Y=NNData[:,5:]
print(X.shape)
print(Y.shape)
print("\n\n")

# Data Normalization
print("maximum dimple angle is", X[:,1:2].max())
print("maximum orientation angle is", X[:,3:4].max())
X[:,1:2]=X[:,1:2]/180
print("maximum normalized dimple angle is", X[:,1:2].max())
X[:,3:4]=X[:,3:4]/180
print("maximum normalized orientation angle is", X[:,3:4].max())

# one-hot encode the developmental stage categorical data
data=X[:,0:1]
data = to_categorical(data)
print(data)
print(data.shape)
X=np.hstack((X,data))
print(X)
print(X.shape)

X=np.delete(X,0,1)
print(X)
print(X.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

scipy.io.savemat('/home/dl2020/Python/BostonHousing/Data.mat', {'X_train':X_train,'Y_train':Y_train,'X_test':X_test,'Y_test':Y_test})
