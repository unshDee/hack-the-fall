import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype

import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
from statsmodels.tsa.arima_model import ARIMA

data = 0
# @cache
def pre_process():
    data = pd.read_csv("L:\hack-the-fall\data\chennai.csv", index_col='Date', parse_dates=True)
    data = data.dropna()

    def ad_test(dataset):
        datatest = adfuller(dataset, autolag='AIC')

    warnings.filterwarnings("ignore")

    train = data.iloc[:-1000]
    test = data.iloc[-1000:]

    return train,test
    

def predict_level(res_name,train):
    model = ARIMA(train[res_name], order=(1, 1, 3))
    model = model.fit()
    pred = model.predict(typ='levels')
    
    # index_future_dates = pd.date_range(start='2020-03-12',end='2020-03-30')

    # pred=model.predict(start=len(data),end=len(data)+18,typ='levels').rename('ARIMA predictions')
    #pred=model.predict(start=0,end=len(data),typ='levels').rename('ARIMA predictions')
  
    # pred=model.predict(typ='levels')

    # pred.index=index_future_dates
    st.write(pred)
    st.line_chart(pred, use_container_width=True)

header = st.container()

dataset = st.container()
sample = st.container()
prediction = st.container()

with header:
    st.title("Aqualarm")
    st.text("In this project we predict the water levels of areas.")

with dataset:
    st.header("The dataset")

    file = st.file_uploader("Upload your dataset here", type={"csv"})
    if file:
        data = pd.read_csv(file)
        st.write(data.head(5))
    else:
        data = pd.read_csv("data/chennai.csv")
        st.subheader("Distribution of water levels of reservoir")

    with st.sidebar:
        st.text("Select columns")
        options = []
        for column in data.columns:
            if is_numeric_dtype(data[column]):
                options.append(column)
        columns = st.multiselect("Columns", options)
        # st.write("You selected:", columns)

    dist = pd.DataFrame(data, columns=columns)
    st.dataframe(dist)
    st.line_chart(dist)

    with prediction:
        st.header("Prediction")
        st.text("Predictions will be present here")
        if st.button('Predict'):
            train,test = pre_process()
            res='POONDI'
            predict_level(res,train)
            
    