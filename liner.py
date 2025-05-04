import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

dataset=pd.read_csv('50_startups.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#missing data
# imputer=SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None) #補平均值
# imputer=imputer.fit(x[:,0:4]) #進行資料擬合,fit範圍
# x[:,0:4]=imputer.transform(x[:,0:4]) #進行缺失資料處理

#categorical data
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])#對第0欄位進行轉換
# labelencoder_y=LabelEncoder()
# y=labelencoder_y.fit_transform(y)

ct=ColumnTransformer([('Country',OneHotEncoder(),[3])],remainder='passthrough') #對第0欄位進行onehoeencoder
X=ct.fit_transform(x)

X=X[:,1:]

#特徵縮放：LinearRegression有自帶特徵縮放
x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=0)
regressor=LinearRegression()# 多元線性回歸
regressor.fit(x_train ,y_train) 
y_pred=regressor.predict(x_test)

X_train=np.append(arr=np.ones((40,1)).astype(int), values=x_train,axis=1)
X_opt=X_train[:,[0,1,2,3,4,5]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=y_train,exog=X_opt).fit()
regressor_OLS.summary()


X_opt=X_train[:,[0,1,3,4,5]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=y_train,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X_train[:,[0,3,4,5]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=y_train,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X_train[:,[0,3,5]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=y_train,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X_train[:,[0,3]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=y_train,exog=X_opt).fit()
regressor_OLS.summary()
