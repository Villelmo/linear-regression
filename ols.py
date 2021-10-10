import statsmodels.api as sm
import pandas as pd 


# reading data from the csv 
data = pd.read_csv('/home/lulu/Descargas/train.csv')

# defining the variables 
x = data['x'].tolist()
y = data['y'].tolist()

# adding the constant term 
x = sm.add_constant(x)

# performing the regression
# and fitting the model 
result = sm.OLS(y,x).fit()

# printing the summary table 
print(result.summary())

