# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:30:40 2022

@author: aarid


"""



import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

"""
Load data from csv file and create a dataframe with only temperature and salinity columns
"""
df = pd.read_csv('bottle.csv')

#print(df.head())

Temp_Salinity = df[['T_degC', 'Salnty']]
Temp_Salinity.columns = ['Temperature', 'Salinity']

"""
Check for and remove null values
"""
missing_values = Temp_Salinity.isnull().sum()
#print(Temp_Salinity.isnull().sum())
Temp_Salinity = Temp_Salinity.dropna(how='any', axis=0)
#print(Temp_Salinity.info())

"""
Plot Temperature Vs. Salinity to see any trends and if linear regression is appropriate
"""
Temp = Temp_Salinity.Temperature
Sal = Temp_Salinity.Salinity

slope, intercept, r_value, p_value, stderr = stats.linregress(Temp, Sal)
r_value_squared = r_value**2
print(r_value_squared)
print(slope)

plt.scatter(Temp, Sal)
plt.show()
plt.clf()

## R squared is 0.255, a very low value. This indicates that a simple linear regression is not appropriate.
## The graph shows a vague possibility of a linear relationship, with outliers.

""" Perform Simple Linear Regression"""

Temp = Temp.values.reshape(-1, 1)

regr = linear_model.LinearRegression()
regr.fit(Temp, Sal)
m = regr.coef_[0]
b = regr.intercept_
print(m, b)

## Slope (m) is negative, however the scatter plot gives the impression that it would be positive

Sal_prediction = regr.predict(Temp)

sns.scatterplot(data=Temp_Salinity, x='Temperature', y='Salinity')
plt.plot(Temp, Sal_prediction, color='red')
plt.show()

## The plot indicates that the linear equation is a poor fit to the data. 


