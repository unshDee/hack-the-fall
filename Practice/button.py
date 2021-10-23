import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
from statsmodels.tsa.arima_model import ARIMA

def predict():
    data = pd.read_csv("L:\hack-the-fall\data\chennai.csv", index_col='Date', parse_dates=True)
    data = data.dropna()

    def ad_test(dataset):
        datatest = adfuller(dataset, autolag='AIC')

    warnings.filterwarnings("ignore")

    train = data.iloc[:-1000]
    test = data.iloc[-1000:]

    model = ARIMA(train['POONDI'], order=(1, 1, 3))
    model = model.fit()

    pred = model.predict(typ='levels')
    st.write(pred)


if st.button('Predict'):
    result = predict()
      
