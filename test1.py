import numpy as np 
import matplotlib.pyplot as mlp
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv('test.csv')

# dataset.head()

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#missing data
imputer=SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None) #補平均值
imputer=imputer.fit(x[:,1:3]) #進行資料擬合,fit範圍
x[:,1:3]=imputer.transform(x[:,1:3]) #進行缺失資料處理

#categorical data
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])#對第0欄位進行轉換
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

ct=ColumnTransformer([('Country',OneHotEncoder(),[0])],remainder='passthrough') #對第0欄位進行onehoeencoder
X=ct.fit_transform(x)

#splitting the Dataset into the Training set and Test set
x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=0)

#Feature Scaling特徵縮放
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


