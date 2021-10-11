# From the implementation point of view, this is just plain Ordinary Least Squares 

import numpy as np 
from sklearn.linear_model import LinearRegression

X = np.array([[1,1],[1,2],[2,2],[2,3]])
y = np.dot(X,np.array([1,2])) + 32

# Fit linear model 
reg = LinearRegression().fit(X,y)

# return the coefficient of determination of the prediction 
print(reg.score(X,y))

# Estimated coefficients for the linear regression problem 
print(reg.coef_)
# Independent term in the linear model. Set to 0.0 if fit_intercept = False
print(reg.intercept_)

# Predict using the linear model 
print(reg.predict(np.array([[3,5]])))
