import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype

header = st.container()

dataset = st.container()
sample = st.container()
prediction, accuracy = st.columns(2)

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
        st.write("You selected:", columns)

    dist = pd.DataFrame(data, columns=columns)
    st.dataframe(dist)
    st.line_chart(dist)

    with prediction:
        st.header("Prediction")
        st.text("Predictions will be present here")

    with accuracy:
        st.header("Testing Accuracy")
        st.text("Accuracy: ")