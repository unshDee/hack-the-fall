#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


# In[3]:


import pandas as pd


# In[5]:


df = pd.read_csv("bengaluru (1).csv")


# In[7]:


df.head()


# In[8]:


df['maxtempC'].plot(figsize=(12,5))


# In[10]:


df['mintempC'].plot(figsize=(12,5))


# In[11]:


df['totalSnow_cm'].plot(figsize=(12,5))


# In[12]:


df['sunHour'].plot(figsize=(12,5))


# In[15]:


plt.scatter('sunHour','totalSnow_cm')
plt.show()


# In[17]:


df.info


# # Cleaning

# In[20]:


df.isnull().sum()


# In[21]:


df.boxplot(figsize=(20,15))
plt.show()


# In[22]:


correlation = df.corr()
sb.heatmap(correlation)


# In[23]:


sb.pairplot(df)


# In[24]:


sb.scatterplot(x= 'pressure', y= 'humidity',hue= 'tempC', data=df)


# In[25]:


sb.scatterplot(x= 'precipMM', y= 'windspeedKmph',hue= 'cloudcover', data=df)


# In[26]:


sb.scatterplot(x= 'maxtempC', y= 'mintempC',hue= 'sunHour', data=df)


# In[27]:


sb.scatterplot(x= 'DewPointC', y= 'FeelsLikeC',hue= 'HeatIndexC', data=df)


# In[ ]:




