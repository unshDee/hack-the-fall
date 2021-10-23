import pandas as pd
import numpy as np

data = pd.read_csv("data\chennai.csv", index_col='Date', parse_dates=True)
data = data.dropna()

from statsmodels.tsa.stattools import adfuller

def ad_test(dataset):
    datatest = adfuller(dataset, autolag='AIC')
    # print("1. ADF : ", datatest[0])
    # print("2. P-Value : ", datatest[1])
    # print("3. : ", datatest[2])
    # print("4. : ", datatest[3])
    # print("5. Critical Values : ")
    # for key, val in datatest[4].items():
        # print("\t", key, ": ", val)
ad_test(data['POONDI'])
ad_test(data['CHOLAVARAM'])
ad_test(data['REDHILLS'])
ad_test(data['CHEMBARAMBAKKAM'])

from pmdarima import auto_arima

import warnings

warnings.filterwarnings("ignore")

stepwise_fit = auto_arima(data['POONDI'], trace=True, supress_warnings=True)
stepwise_fit.summary()

stepwise_fit = auto_arima(data['CHOLAVARAM'], trace=True, supress_warnings=True)
stepwise_fit.summary()

stepwise_fit = auto_arima(data['REDHILLS'], trace=True, supress_warnings=True)
stepwise_fit.summary()

stepwise_fit = auto_arima(data['CHEMBARAMBAKKAM'], trace=True, supress_warnings=True)
stepwise_fit.summary()


from statsmodels.tsa.arima_model import ARIMA

train = data.iloc[:-1000]
test = data.iloc[-1000:]

model = ARIMA(train['POONDI'], order=(1, 1, 3))
model = model.fit()
model.summary()
data.tail()

pred = model.predict(typ='levels')
print(pred)

index_future_dates = pd.date_range(start='2020-03-12', end='2020-03-14')
# print(index_future_dates)
pred = model.predict(start=len(data), end=len(data) + 2, typ='levels').rename('ARIMA predictions')
# print(comp_pred)
pred.index = index_future_dates
print(pred)

pred.plot(figsize=(12, 5))


model1 = ARIMA(data['CHOLAVARAM'], order=(4, 1, 2))
model1 = model1.fit()
model1.summary()
data.tail()

index_future_dates = pd.date_range(start='2020-03-12', end='2020-03-14')
# print(index_future_dates)
pred = model1.predict(start=len(data), end=len(data) + 2, typ='levels').rename('ARIMA predictions')
# print(comp_pred)
pred.index = index_future_dates
print(pred)


model2 = ARIMA(data['REDHILLS'], order=(1, 1, 2))
model2 = model2.fit()
model2.summary()
data.tail()

index_future_dates = pd.date_range(start='2020-03-12', end='2020-03-14')
# print(index_future_dates)
pred = model2.predict(start=len(data), end=len(data) + 2, typ='levels').rename('ARIMA predictions')
# print(comp_pred)
pred.index = index_future_dates
print(pred)


model3 = ARIMA(data['CHEMBARAMBAKKAM'], order=(1, 1, 2))
model3 = model3.fit()
model3.summary()
data.tail()

index_future_dates = pd.date_range(start='2020-03-12', end='2020-03-14')
# print(index_future_dates)
pred = model3.predict(start=len(data), end=len(data) + 2, typ='levels').rename('ARIMA predictions')
# print(comp_pred)
pred.index = index_future_dates
print(pred)
