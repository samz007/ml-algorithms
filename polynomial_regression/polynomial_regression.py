import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# fitting part  liear regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X, y)

# fitting part of polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)

linear_reg_two = LinearRegression()
linear_reg_two.fit(x_poly, y)

# predict when input is 6.5
print(linear_reg_two.predict(poly_reg.fit_transform([[6.5]])))

# visualization
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, linear_reg_two.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.show()
