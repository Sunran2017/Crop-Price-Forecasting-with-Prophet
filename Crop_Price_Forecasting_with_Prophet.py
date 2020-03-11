
# coding: utf-8

# In[4]:


######corn price####


import pandas as pd
from fbprophet import Prophet
from datetime import datetime
from fbprophet.plot import add_changepoints_to_plot
df = pd.read_csv("corn_price.csv")
df.columns=["ds","y"]


#prediction periods#
future = m.make_future_dataframe(periods=3650*3+47)
future.tail()

##yearly seasonality, changepoint_prior_scale = 0.6#
m = Prophet(changepoint_prior_scale=0.6,weekly_seasonality=False)
m.add_seasonality(name='yearly', period=365, fourier_order=30)
forecast = m.fit(df).predict(future)
forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail()

#data save to csv #
yhat = forecast[["ds","yhat"]]
yhat = yhat.set_index('ds')
corn_price_predict = yhat.resample('AS').mean()
corn_price_predict.to_csv('corn_price.csv')

#figure#
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(),m,forecast)
fig = m.plot_components(forecast)


# In[ ]:


######soy price####


import pandas as pd
from fbprophet import Prophet
from datetime import datetime
from fbprophet.plot import add_changepoints_to_plot
df = pd.read_csv("soy_price.csv")
df.columns=["ds","y"]


#prediction periods#
future = m.make_future_dataframe(periods=3650*3+47)
future.tail()

##yearly seasonality, changepoint_prior_scale = 0.6#
m = Prophet(changepoint_prior_scale=0.6,weekly_seasonality=False)
m.add_seasonality(name='yearly', period=365, fourier_order=30)
forecast = m.fit(df).predict(future)
forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail()

#data save to csv #
yhat = forecast[["ds","yhat"]]
yhat = yhat.set_index('ds')
soy_price_predict = yhat.resample('AS').mean()
soy_price_predict.to_csv('soy_price.csv')

#figure#
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(),m,forecast)
fig = m.plot_components(forecast)


# In[ ]:


######wheat price####


import pandas as pd
from fbprophet import Prophet
from datetime import datetime
from fbprophet.plot import add_changepoints_to_plot
df = pd.read_csv("wheat_price.csv")
df.columns=["ds","y"]


#prediction periods#
future = m.make_future_dataframe(periods=3650*3+47)
future.tail()

##yearly seasonality, changepoint_prior_scale = 0.6#
m = Prophet(changepoint_prior_scale=0.6,weekly_seasonality=False)
m.add_seasonality(name='yearly', period=365, fourier_order=30)
forecast = m.fit(df).predict(future)
forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail()

#data save to csv #
yhat = forecast[["ds","yhat"]]
yhat = yhat.set_index('ds')
wheat_price_predict = yhat.resample('AS').mean()
wheat_price_predict.to_csv('wheat_price.csv')

#figure#
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(),m,forecast)
fig = m.plot_components(forecast)

