import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# fitting part  random forest regression
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

# prediction
y_pred = regressor.predict([[6.5]])
print(y_pred)

# visualization
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict((X_grid)), color='blue')
plt.show()
