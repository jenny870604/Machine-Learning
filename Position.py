import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset=pd.read_csv('Postition_Salarites.csv')

x=dataset.iloc[:,[1]].values
y=dataset.iloc[:,2].values

lin_reg=LinearRegression()
lin_reg.fit(x, y)


plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

poly_reg=PolynomialFeatures(degree=4)
X_ploy=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_ploy, y)


plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

X_grid=np.arange(min(x),max(x), 0.1)
X_grid=X_grid.reshape(len(X_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

new_x=6.5
new_x=np.array(new_x).reshape(-1, 1)
lin_reg.predict(new_x)#線性回歸
lin_reg_2.predict(poly_reg.fit_transform(new_x))#多項式回歸

