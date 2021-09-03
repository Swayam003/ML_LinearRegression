import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet,ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
#from pandas_profiling import ProfileReport
import pickle
#import matplotlib.pyplot as plt
#%matplotlib inline

def building_model():
#EDA
    df = pd.read_csv("ai4i2020.csv")
    df = df.drop(columns = ['UDI','Product ID','Type'])
    y = df['Air temperature [K]']
    x = df.drop( columns=['Air temperature [K]','Rotational speed [rpm]'] )
# Scalling
    scalar = StandardScaler()
    arr = scalar.fit_transform(x)
#Splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(arr,y,test_size= 0.20, random_state = 100)
# Building model
    linear = LinearRegression()
    linear.fit(x_train,y_train)
    print(f"Accurarcy while training the data {linear.score(x_train,y_train)} \n")
    print(f"Accurarcy while testing the data {linear.score(x_test,y_test)} \n")
#saving Linear regression Model in a file
    filename = "machine_model.sav"
    pickle.dump(linear,open(filename,'wb'))
    return x, y
def standardize(c):
    """
    This function converts inputted data into Standarized form
    :param c: Pass the data(in the form of 2-d array) through which model will make the prediction
    :return: It will return Standardized data
    """
    x, y = building_model()
    scalar = StandardScaler()
    data = scalar.fit_transform(x,y)
    return data
