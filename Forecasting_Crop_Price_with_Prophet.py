#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:23:27 2020

@author: ran
"""

#!pip install fbprophet --upgrade

import pandas as pd
from fbprophet import Prophet
from datetime import datetime
from fbprophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
import numpy as np


dataPath = "/Users/ran/ABM/cropPrice/"
resultPath = "/Users/ran/ABM/cropPrice/results/"
cornFile = "cornPrice.csv"
soyFile = "soyPrice.csv"
wheatFile = "wheatPrice.csv"

# corn 

 #input csv file 
def readCrop(file):   
    data = dataPath + file
    df = pd.read_csv(data)
    df.columns=["ds","y"]
    df.set_index('ds').y.plot().figure
    return df;

df = readCrop(cornFile)


#Initializing
m = Prophet()
m.fit(df)

term = 34;

#prediction periods#
if term == 34 :
    future = m.make_future_dataframe(periods = 3650 * 3 + 14)
elif term ==10:
    future = m.make_future_dataframe(periods = 365 * 10)
elif term == 5:
    future = m.make_future_dataframe(periods = 365 * 5)
elif term == 3:
    future = m.make_future_dataframe(periods = 365 * 3)
        
        

##yearly seasonality, changepoint_prior_scale = 0.8#
m = Prophet(changepoint_prior_scale = 0.8, weekly_seasonality = False, daily_seasonality=False)
m.add_seasonality(name='yearly', period = 365 * 10, fourier_order = 30)


#forecast
forecast = m.fit(df).predict(future)
forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail()
yhat = forecast[["ds","yhat"]]


###evaluate performance

def make_comparison_dataframe(historical, forecast):
    return yhat.set_index("ds")[["yhat"]].join(historical.set_index("ds"))


##define a function to calculate MAPE and MAE
def calculate_forecast_errors(df, prediction_size):
    df = df.copy()
    df["e"] = df["y"] - df["yhat"]
    df["p"] = 100 * df["e"] / df["y"]
    predicted_part = df[-prediction_size:]
    error_mean = lambda error_name:np.mean(np.abs(predicted_part[error_name]))
    return {"MAPE":error_mean('p'), "MAE": error_mean('e')}


cmp_df = make_comparison_dataframe(df, yhat)[0:1456]
prediction_size = 54 * 8  # 8 years prediction
for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
    print (err_name,err_value)
    


#Results

#save figure
def saveFig(name, yr):
    fig = m.plot(forecast)
    plt.xlabel("year")
    plt.ylabel("CAD")
    plt.title(name + " price forecast, weekly prices from " + str(yr) + " to 2023")
    plt.subplots_adjust(top = 0.9)
    plt.savefig(resultPath + name + str(term) + "1.png",dpi = 800)
    a = add_changepoints_to_plot(fig.gca(),m,forecast) 
    fig = m.plot_components(forecast)
    plt.savefig(resultPath + name + str(term) + "2.png")

saveFig("Corn", 1992)

    
#save forcasting results to csv file

yhat = yhat.set_index("ds")

def saveFcst(name):
    fcst = yhat.resample('AS').mean();
    fcst.to_csv(resultPath + name + str(term) + "Price_fcst.csv")

saveFcst("corn")

############################################################################

# soy

df = readCrop(soyFile)
term = 3;
name = "Soybean";

#Initializing
m = Prophet()
m.fit(df)

#prediction periods#
if term == 34 :
    future = m.make_future_dataframe(periods = 3650 * 3 + 14)
elif term ==10:
    future = m.make_future_dataframe(periods = 365 * 10)
elif term == 5:
    future = m.make_future_dataframe(periods = 365 * 5)
elif term == 3:
    future = m.make_future_dataframe(periods = 365 * 3)
        
##yearly seasonality, changepoint_prior_scale = 0.8#
m = Prophet(changepoint_prior_scale = 0.8,weekly_seasonality = False,daily_seasonality=False)
m.add_seasonality(name='yearly', period = 365, fourier_order = 30)


#forecast
forecast = m.fit(df).predict(future)
forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail()
yhat = forecast[["ds","yhat"]]


# evaluation performance for prediction
cmp_df = make_comparison_dataframe(df, yhat)[0:1144]

prediction_size = 54 * 6  # 6 years prediction
for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
    print (err_name,err_value)
    
#Results
saveFig(name, 1998)
#save 
yhat = yhat.set_index("ds")
saveFcst(name)


############################################################################

#Wheat

df = readCrop(wheatFile)
term = 34;
name = "Wheat";
#Initializing
m = Prophet()
m.fit(df)

#prediction periods#
if term == 34 :
    future = m.make_future_dataframe(periods = 3650 * 3 + 14)
elif term ==10:
    future = m.make_future_dataframe(periods = 365 * 10)
elif term == 5:
    future = m.make_future_dataframe(periods = 365 * 5)
elif term == 3:
    future = m.make_future_dataframe(periods = 365 * 3)

#yearly seasonality,changepoint_prior_scale=1.7
m = Prophet(changepoint_prior_scale = 1.7,weekly_seasonality = False)
m.add_seasonality(name='yearly', period=365 * 4, fourier_order = 30)
forecast = m.fit(df).predict(future)
forecast[["ds","yhat","yhat_lower","yhat_upper"]]
yhat = forecast[["ds","yhat"]]

# evaluation performance for prediction
cmp_df = make_comparison_dataframe(df, yhat)[0:832]

prediction_size = 54*4  # 4 years prediction
for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
    print (err_name,err_value)


#save fig
saveFig(name, 2004)
#save 
yhat = yhat.set_index("ds")
saveFcst(name)

############################################################################
# soy to corn ratio price

df0 = readCrop(cornFile)
df0 = df0[312:]
df = readCrop(soyFile).merge(df0, how='inner', on='ds')
df.columns = ["ds", "otherCrop","baseCorn"]


df["y"] = df["otherCrop"]/df["baseCorn"]
df = df.drop(["otherCrop","baseCorn"], axis = 1)
df.set_index('ds').y.plot().figure


term = 3;
name = "Soybean to corn ratio";

#Initializing
m = Prophet()
m.fit(df)

#prediction periods#
if term == 34 :
    future = m.make_future_dataframe(periods = 3650 * 3 + 14)
elif term ==10:
    future = m.make_future_dataframe(periods = 365 * 10)
elif term == 5:
    future = m.make_future_dataframe(periods = 365 * 5)
elif term == 3:
    future = m.make_future_dataframe(periods = 365 * 3)
        
##yearly seasonality, changepoint_prior_scale = 0.8#
m = Prophet(changepoint_prior_scale = 0.8, weekly_seasonality = False,daily_seasonality=False)
m.add_seasonality(name='yearly', period = 365 * 10, fourier_order = 30)


#forecast
forecast = m.fit(df).predict(future)
forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail()
yhat = forecast[["ds","yhat"]]


# evaluation performance for prediction
cmp_df = make_comparison_dataframe(df, yhat)[0:1144]

prediction_size = 54 * 6  # 6 years prediction
for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
    print (err_name,err_value)
    
#Results
saveFig(name, 1998)
#save 
yhat = yhat.set_index("ds")
saveFcst(name)


############################################################################


# wheat to corn ratio price


df0 = readCrop(cornFile)
df0 = df0[624:]
df = readCrop(wheatFile).merge(df0, how='inner', on='ds')
df.columns = ["ds", "otherCrop","baseCorn"]



df["y"] = df["otherCrop"]/df["baseCorn"]
df = df.drop(["otherCrop","baseCorn"], axis = 1)
df.set_index('ds').y.plot().figure


term = 3;
name = "Wheat to corn ratio";

#Initializing
m = Prophet()
m.fit(df)

#prediction periods#
if term == 34 :
    future = m.make_future_dataframe(periods = 3650 * 3 + 14)
elif term ==10:
    future = m.make_future_dataframe(periods = 365 * 10)
elif term == 5:
    future = m.make_future_dataframe(periods = 365 * 5)
elif term == 3:
    future = m.make_future_dataframe(periods = 365 * 3)
        

#yearly seasonality,changepoint_prior_scale=1.7
m = Prophet(changepoint_prior_scale = 1.7,weekly_seasonality = False)
m.add_seasonality(name='yearly', period= 365 * 4, fourier_order = 30)
forecast = m.fit(df).predict(future)
forecast[["ds","yhat","yhat_lower","yhat_upper"]]
yhat = forecast[["ds","yhat"]]

# evaluation performance for prediction
cmp_df = make_comparison_dataframe(df, yhat)[0:832]

prediction_size = 54 * 4  # 4 years prediction
for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
    print (err_name,err_value)


#save fig
saveFig(name, 2004)
#save 
yhat = yhat.set_index("ds")
saveFcst(name)


