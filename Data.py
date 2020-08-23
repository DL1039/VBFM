import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import to_categorical

import pandas as pd
import seaborn as sb
from pylab import rcParams
from pandas import DataFrame

rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')

#Load NN Data from mat file into dict.
mat=scipy.io.loadmat(r'/home/dl2020/Python/NeuralNetwork/NNData.mat')
print(mat.keys())

#returns the NNData value forom dict. 
NNData=mat["NNData"]
print(NNData.shape)

#convert list to panda dataframe
df=DataFrame(NNData)
df.columns=['dev._stage','dimple_ang.','radii_ratio','orientation_ang.','area','force']
print(df.head(10))

#Checking for missing values
print(df.isnull().sum())

#print data information
print(df.info())

#Converting categorical variables to a dummy indicators
stage=pd.get_dummies(df['dev._stage'],drop_first=False)
print(stage.head(10))
df.drop(['dev._stage'],axis=1,inplace=True)
df=pd.concat([df,stage],axis=1)
print(df.head())

#rename new columns
df.rename(columns = {1:'ds1'}, inplace = True) 
df.rename(columns = {2:'ds2'}, inplace = True)
df.rename(columns = {3:'ds3'}, inplace = True)

#change order of columns
Col_Order = ['dimple_ang.','radii_ratio','orientation_ang.','area','ds1','ds2','ds3','force']
df = df.reindex(columns=Col_Order)
print(df.head())

#Checking for independence between features
sb.heatmap(df.corr(), annot=True)
plt.show()

#data normalization using mean normalization
#df['dimple_ang.']=(df['dimple_ang.']-df['dimple_ang.'].mean())/df['dimple_ang.'].std()
#df['orientation_ang.']=(df['orientation_ang.']-df['orientation_ang.'].mean())/df['orientation_ang.'].std()
#print(df.head())

#data normalization using min-max normalization
df['dimple_ang.']=(df['dimple_ang.']-df['dimple_ang.'].min())/(df['dimple_ang.'].max()-df['dimple_ang.'].min())
df['orientation_ang.']=(df['orientation_ang.']-df['orientation_ang.'].min())/(df['orientation_ang.'].max()-df['orientation_ang.'].min())
print(df.head())


X = df.iloc[:,0:7]
y = df.iloc[:,7:8]

#######################################Implementation with List########################
# X=NNData[:,0:5]
# Y=NNData[:,5:]
# print(X.shape)
# print(Y.shape)
# print("\n\n")

# #Data Normalization
# print("maximum dimple angle is", X[:,1:2].max())
# print("maximum orientation angle is", X[:,3:4].max())
# X[:,1:2]=X[:,1:2]/180
# print("maximum normalized dimple angle is", X[:,1:2].max())
# X[:,3:4]=X[:,3:4]/180
# print("maximum normalized orientation angle is", X[:,3:4].max())

# # one-hot encode the developmental stage categorical data
# data=X[:,0:1]
# data = to_categorical(data)
# print(data)
# print(data.shape)
# X=np.hstack((X,data))
# print(X)
# print(X.shape)

# X=np.delete(X,0,1)
# print(X)
# print(X.shape)
#######################################Implementation with List########################

#split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#converting dataframe to list
X_train_l = X_train.values.tolist()
y_train_l = y_train.values.tolist()
X_test_l = X_test.values.tolist()
y_test_l = y_test.values.tolist()

#save data into mat file
scipy.io.savemat('/home/dl2020/Python/BostonHousing/Data.mat', {'X_train':X_train_l,'y_train':y_train_l,'X_test':X_test_l,'y_test':y_test_l})
