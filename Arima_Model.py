#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# In[2]:


data = pd.read_csv("chennai_reservoir_levels.csv", index_col='Date', parse_dates=True)

# In[3]:


data.head()

# In[4]:


data.shape

# In[5]:


data = data.dropna()

# In[6]:


data.shape

# In[7]:


data['POONDI'].plot(figsize=(12, 5))

# In[8]:


data['CHOLAVARAM'].plot(figsize=(12, 5))

# In[9]:


data['REDHILLS'].plot(figsize=(12, 5))

# In[10]:


data['CHEMBARAMBAKKAM'].plot(figsize=(12, 5))

# In[11]:


from statsmodels.tsa.stattools import adfuller


def ad_test(dataset):
    datatest = adfuller(dataset, autolag='AIC')
    print("1. ADF : ", datatest[0])
    print("2. P-Value : ", datatest[1])
    print("3. : ", datatest[2])
    print("4. : ", datatest[3])
    print("5. Critical Values : ")
    for key, val in datatest[4].items():
        print("\t", key, ": ", val)


# In[12]:


ad_test(data['POONDI'])

# In[13]:


ad_test(data['CHOLAVARAM'])

# In[14]:


ad_test(data['REDHILLS'])

# In[15]:


ad_test(data['CHEMBARAMBAKKAM'])

# In[16]:


from pmdarima import auto_arima

import warnings

warnings.filterwarnings("ignore")

# In[17]:


stepwise_fit = auto_arima(data['POONDI'], trace=True, supress_warnings=True)

stepwise_fit.summary()

# In[18]:


stepwise_fit = auto_arima(data['CHOLAVARAM'], trace=True, supress_warnings=True)

stepwise_fit.summary()

# In[19]:


stepwise_fit = auto_arima(data['REDHILLS'], trace=True, supress_warnings=True)

stepwise_fit.summary()

# In[20]:


stepwise_fit = auto_arima(data['CHEMBARAMBAKKAM'], trace=True, supress_warnings=True)

stepwise_fit.summary()

# In[21]:


from statsmodels.tsa.arima_model import ARIMA

# In[22]:


print(data.shape)

# In[23]:


train = data.iloc[:-1000]
test = data.iloc[-1000:]
print(train.shape, test.shape)

# In[24]:


model = ARIMA(train['POONDI'], order=(1, 1, 3))
model = model.fit()
model.summary()

# In[25]:


pred = model.predict(typ='levels')
print(pred)

# In[26]:


model = ARIMA(data['POONDI'], order=(1, 1, 3))
model = model.fit()
model.summary()
data.tail()

# In[27]:


index_future_dates = pd.date_range(start='2020-03-12', end='2020-03-14')
# print(index_future_dates)
pred = model.predict(start=len(data), end=len(data) + 2, typ='levels').rename('ARIMA predictions')
# print(comp_pred)
pred.index = index_future_dates
print(pred)

# In[28]:


pred.plot(figsize=(12, 5))

# In[29]:


model1 = ARIMA(data['CHOLAVARAM'], order=(4, 1, 2))
model1 = model1.fit()
model1.summary()
data.tail()

# In[30]:


index_future_dates = pd.date_range(start='2020-03-12', end='2020-03-14')
# print(index_future_dates)
pred = model1.predict(start=len(data), end=len(data) + 2, typ='levels').rename('ARIMA predictions')
# print(comp_pred)
pred.index = index_future_dates
print(pred)

# In[32]:


model2 = ARIMA(data['REDHILLS'], order=(1, 1, 2))
model2 = model2.fit()
model2.summary()
data.tail()

# In[33]:


index_future_dates = pd.date_range(start='2020-03-12', end='2020-03-14')
# print(index_future_dates)
pred = model2.predict(start=len(data), end=len(data) + 2, typ='levels').rename('ARIMA predictions')
# print(comp_pred)
pred.index = index_future_dates
print(pred)

# In[34]:


model3 = ARIMA(data['CHEMBARAMBAKKAM'], order=(1, 1, 2))
model3 = model3.fit()
model3.summary()
data.tail()

# In[35]:


index_future_dates = pd.date_range(start='2020-03-12', end='2020-03-14')
# print(index_future_dates)
pred = model3.predict(start=len(data), end=len(data) + 2, typ='levels').rename('ARIMA predictions')
# print(comp_pred)
pred.index = index_future_dates
print(pred)

# In[ ]:
