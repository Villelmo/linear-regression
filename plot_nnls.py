""" We fit a linear model with positive constraints on the regression 
    coefficients and compare the estimaded coefficients
    to a classic linear regression 
"""

print(__doc__)
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score

# Generate some random data 

np.random.seed(42)

n_samples, n_features = 200, 50
X = np.random.randn(n_samples,n_features)
true_coef = 3 * np.random.randn(n_features)

# Threshold coefficients to render them non-negative
true_coef[true_coef < 0] = 0
y = np.dot(X,true_coef)


# add some noise 
y += 5 * np.random.normal(size=(n_samples, ))

# split the data in train set and test set 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)


# Fit the non-negative least squares 
from sklearn.linear_model import LinearRegression

reg_nnls = LinearRegression(positive=True)
y_pred_nnls = reg_nnls.fit(X_train,y_train).predict(X_test)
r2_score_nnls = r2_score(y_test,y_pred_nnls)
print("NNLS R2 score",r2_score_nnls)

# Fit an OLS 
reg_ols = LinearRegression()
y_pred_ols = reg_ols.fit(X_train,y_train).predict(X_test)
r2_score_ols = r2_score(y_test,y_pred_ols)
print("OLS R2 score",r2_score_ols)

# Comparing the regression coefficients between OLS and NNLS 
fig, ax = plt.subplots()
ax.plot(reg_ols.coef_,reg_nnls.coef_,linewidth=0,marker=".")

low_x,high_x = ax.get_xlim()
low_y,high_y = ax.get_ylim()
low = max(low_x,low_y)
high = min(high_x,high_y)
ax.plot([low,high],[low,high],ls="--",c=".3",alpha=.5)
ax.set_xlabel("OLS regression coefficients",fontweight="bold")
ax.set_ylabel("NNLS regression coefficients",fontweight="bold")
