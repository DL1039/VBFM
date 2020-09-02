
from keras.models import load_model
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from datetime import datetime
import os

mydir = os.path.join(os.getcwd(), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)

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

model = load_model('/home/dl2020/Python/ML/projects/VBFM/VBFM2.h5')

loss=model.evaluate(X_train,y_train)
print('Train loss:',loss)
#print('Train acc:',acc)

loss =model.evaluate(X_test,y_test)
print('Test loss:',loss)
# print('Test acc:',acc)

predicted_y_train=model.predict(X_train)
plt.title('Train Data')
plt.plot(X_train[:,0:1], y_train, 'ro', X_train[:,0:1],predicted_y_train,'bs',markersize=1)
#plt.plot(X_train[:,0:1],predicted_y_train,'bs',markersize=1)
#plt.show()
plt.savefig('Train.png')

predicted_y_test=model.predict(X_test)
plt.title('Test Data')
plt.plot(X_test[:,0:1], y_test, 'ro', X_test[:,0:1],predicted_y_test,'bs',markersize=1)
#plt.show()
plt.savefig('Test.png')

# digit = model.predict(X_test[130:131,:])
# print(X_test[130:131,:])
# print(digit)
# print(y_test[130:131,:])
