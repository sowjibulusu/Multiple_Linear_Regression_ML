#Multi Linear Regression
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pa

#Importing the Dataset
dataset = pa.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding Ctegorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_x = LabelEncoder()
x[:, 3] = label_encoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()  

#Avoiding Dummy variable trap
x=x[:, 1:]
#Splitting the dataset into Training Set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

'''#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
##''For Training Set we need to fit and Transform where as for test set we
##only need to transform, because it is already fitted to training set

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''

#Fitting Multi Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(x_test)

#Building the optimal model using backend elimination
import statsmodels.api as sm
'''statsmodels library is mainly used to compute the P values and 
evaluate the Statistical Significance of the Independent Variables'''
'''In the x, we dont have the constant b0 column as per the multilinear regression
general eqation...Hence we need to add that to the dataset-->For this we just 
need to add column of 1 in the variable x'''
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
#Backward Elimination Starts
x_optimal = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()
x_optimal = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()
x_optimal = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()
x_optimal = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()
x_optimal = x[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()


