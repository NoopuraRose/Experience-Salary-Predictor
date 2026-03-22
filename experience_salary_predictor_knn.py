# library imports
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# data set reading
mydata = pd.read_csv("experience_salary.csv")
x = mydata[["X"]]
y = mydata[["Y"]]

# model creation
model = KNeighborsRegressor(n_neighbors = 3)
model.fit(x,y)

# predicting new value
new_salary = model.predict([[160]])
print("Predicted salary = ", new_salary)

# model evaluation
y_pred = model.predict(x)
mse = mean_squared_error(y,y_pred)
print("MSE = ", mse)
rmse = np.sqrt(mse)
print("RMSE = ", rmse)

# visualization
plt.scatter(x,y)
plt.plot(x,y_pred, color = 'red')
plt.show()