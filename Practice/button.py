import streamlit as st

def add(a,b):
    c = a+b
    return c

st.write('Enter two numbers to add')

num1 = st.number_input('insert number 1')
num2 = st.number_input('insert number 2')

if st.button('ADD'):
    result = add(num1,num2)
    st.write('The Addition of', num1, 'and', num2,'is equal to', result, '.')
