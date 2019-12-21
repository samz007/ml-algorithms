import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# we scale features for training set
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# building svr
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# predictions
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
sc_y.inverse_transform(y_pred)
print(sc_y.inverse_transform(y_pred))

# visualization
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.show()
