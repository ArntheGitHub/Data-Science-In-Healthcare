import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os

# Assuming you have the 'df' DataFrame available with the correct data.
# Load the CSV files to df
df = pd.read_csv('stroke_data.csv')  # Adjust the file name as necessary

st.markdown("""
# Welcome to Our Stroke Prediction Application 🏥

## Project Introduction
Welcome! This application aims to predict the likelihood of a brain stroke based on a variety of health and lifestyle factors. Our analysis is based on a dataset that includes key indicators related to stroke occurrences.


""")

st.write("### Data Overview", df.head())

# Text input for the search query
search_query = st.text_input("Enter a name to search:")

# Display the full DataFrame if the search query is empty
if search_query == "":
    st.dataframe(df.head())  # Show only the first few rows using .head()
else:
    # Filter the DataFrame based on the search query in a relevant column (e.g., 'gender' or 'age')
    filtered_df = df[df['gender'].str.contains(search_query, case=False, na=False)]  # Adjust column as necessary

    # Display the filtered DataFrame
    st.dataframe(filtered_df)

# Load  model
model = load('stroke_prediction_model.pkl')  # Ensure you have a model trained for stroke prediction

# Define the layout of your app
st.title('Stroke Prediction App')

# Create user input fields
st.header('Enter your details:')
age = st.number_input('Age', min_value=0)
hypertension = st.number_input('Hypertension (0 for No, 1 for Yes)', min_value=0, max_value=1)
heart_disease = st.number_input('Heart Disease (0 for No, 1 for Yes)', min_value=0, max_value=1)
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, format="%.2f")
bmi = st.number_input('BMI', min_value=0.0, format="%.2f")
gender_options = ['Male', 'Female', 'Other']  # Update or adjust as necessary
gender = st.selectbox('Gender', gender_options)
smoking_status_options = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']  # Adjust based on your model
smoking_status = st.selectbox('Smoking Status', smoking_status_options)

# Predict stroke likelihood
if st.button('Predict Stroke Likelihood'):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([[
        age,
        hypertension,
        heart_disease,
        avg_glucose_level,
        bmi,
        gender,
        smoking_status
    ]], columns=[
        'age',
        'hypertension',
        'heart_disease',
        'avg_glucose_level',
        'bmi',
        'gender',
        'smoking_status'
    ])
    
    # Get the prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    prediction_text = "Likely" if prediction[0] == 1 else "Unlikely"
    st.write(f"The prediction is: Stroke is {prediction_text}")
