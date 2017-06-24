# predicting body weights from brain weights - supervised learning
# read our dataset
import pandas as pd
# we are using linear regression
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data
dataframe = pd.read_fwf('brain_body.txt') # 2d arrangement of rows and cols
#print(dataframe)

x_values, y_values = dataframe[['Brain']], dataframe[['Body']]

# train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()