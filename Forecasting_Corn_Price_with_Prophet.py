
# coding: utf-8

#import pkg

import pandas as pd
from fbprophet import Prophet
from datetime import datetime
from fbprophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
import numpy as np

#read csv file
df = pd.read_csv("corn_price.csv")
df.columns=["ds","y"]
df.head()


#Initializing
m = Prophet()
m.fit(df)

#prediction periods#
future = m.make_future_dataframe(periods=3650*3+14)
future.tail()

#yearly seasonality, changepoint_prior_scale = 0.8
m = Prophet(changepoint_prior_scale=0.8,weekly_seasonality=False,daily_seasonality=False)
m.add_seasonality(name='yearly', period=365, fourier_order=30)
forecast = m.fit(df).predict(future)


forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail()
yhat = forecast[["ds","yhat"]]

#evaluate performance

def make_comparison_dataframe(historical, forecast):
    return yhat.set_index("ds")[["yhat"]].join(historical.set_index("ds"))

cmp_df = make_comparison_dataframe(df, yhat)[0:1456]

##define a function to calculate MAPE and MAE
def calculate_forecast_errors(df, prediction_size):
    
    df = df.copy()
    
    df["e"] = df["y"] - df["yhat"]
    df["p"] = 100 * df["e"] / df["y"]
    
    predicted_part = df[-prediction_size:]
    
    error_mean = lambda error_name:np.mean(np.abs(predicted_part[error_name]))
    
    return {"MAPE":error_mean('p'), "MAE": error_mean('e')}

prediction_size = 54*8  # 30% for prediction

for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
    print (err_name,err_value)


#save forcasting results to csv file
yhat = yhat.set_index("ds")
corn_price_predict = yhat.resample('AS').mean()
corn_price_predict.to_csv('corn_price_fcst.csv')

#generate figure
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(),m,forecast) 
fig = m.plot_components(forecast)

