import streamlit as st
import pickle
import pandas as pd
import numpy as np

with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
st.title('Linear Regression Model')

# Input fields for the features
TV = st.number_input('TV Advertising Budget', min_value=0.0, step=0.1)
Radio = st.number_input('Radio Advertising Budget', min_value=0.0, step=0.1)
Newspaper = st.number_input('Newspaper Advertising Budget', min_value=0.0, step=0.1)

# Predict button
if st.button('Predict Sales'):
    input_data = np.array([[TV, Radio, Newspaper]])
    prediction = model.predict(input_data)
    st.write(f'Predicted Sales: {prediction}')
    st.write(f'Predicted Sales: {prediction[0]:.2f}')

