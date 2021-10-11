from sklearn import linear_model

reg = linear_model.LinearRegression()

# LinearRegression will take in its fit method arrays X
# y and will store the coefficients w of the linear model in its coef_ member
print(reg.fit([[0,0],[1,1],[2,2]],[0,1,2]))
# The coefficient estimates for Ordinary Least Squares rely on the independence of the features 
print(reg.coef_)
