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
# NNData=mat["NNData"]
# print(NNData.shape)

#convert list to panda dataframe
#df=DataFrame(NNData)

TrainData=mat["TrainData"]
ValidData=mat["ValidData"]
TestData=mat["TestData"]
print(TestData.shape)

df=pd.concat([DataFrame(TrainData),DataFrame(ValidData),DataFrame(TestData)],axis=0)
print(df.shape)

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

# Plotting the heatmap of correlation between features for checking the dependency between features
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

# Viewing the data statistics
print(df.describe())

X_train = df.iloc[0:42004,0:7]
y_train = df.iloc[0:42004,7:8]
X_test = df.iloc[42004:,0:7]
y_test = df.iloc[42004:,7:8]

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
scipy.io.savemat('/home/dl2020/Python/BostonHousing/Data2.mat', {'X_train':X_train_l,'y_train':y_train_l,'X_test':X_test_l,'y_test':y_test_l})
