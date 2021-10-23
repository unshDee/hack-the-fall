import pandas as pd
import numpy as np
import streamlit as st

data = pd.read_csv("data\chennai.csv", index_col='Date', parse_dates=True)
data = data.dropna()

from statsmodels.tsa.stattools import adfuller

def ad_test(dataset):
    datatest = adfuller(dataset, autolag='AIC')


from pmdarima import auto_arima

import warnings

warnings.filterwarnings("ignore")

from statsmodels.tsa.arima_model import ARIMA

train = data.iloc[:-1000]
test = data.iloc[-1000:]

model = ARIMA(train['POONDI'], order=(1, 1, 3))
model = model.fit()


pred = model.predict(typ='levels')
st.write(pred)
print(pred)

# index_future_dates = pd.date_range(start='2020-03-12', end='2020-03-14')
# # print(index_future_dates)
# pred = model.predict(start=len(data), end=len(data) + 2, typ='levels').rename('ARIMA predictions')
# # print(comp_pred)
# pred.index = index_future_dates


