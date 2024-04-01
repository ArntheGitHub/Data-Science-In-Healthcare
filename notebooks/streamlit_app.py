import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Load the CSV files to df
df = pd.read_csv('data\cleaned\dataset_encoded.csv')
df = pd.read_csv('data\cleaned\stroke_data_cleaned.csv')
df = df.dropna()

# Load the trained model and associated data
with open("notebooks\Logistic_Regression.pkl", "rb") as f:
    trained_data = pickle.load(f)

# Extract the model and other data
model = trained_data['model']
X_train = trained_data['X_train']


# Function to create the interactive scatter plot
def create_interactive_scatter_plot(df, target, features):
    st.title('Interactive Scatter Plot')

    # Sidebar for user input:
    # User can select the x-axis for the scatter plot from features list
    x_axis_options = features
    x_axis = st.sidebar.selectbox('Select the x-axis for the scatter plot:', x_axis_options)

    # Create a Plotly Express scatter plot
    fig = px.scatter(df, x=x_axis, y=target,
                     color=target, hover_data=features)
    
    # Update layout if desired
    fig.update_layout(
        title=f'Scatter Plot of {target.capitalize()} vs. {x_axis}',
        xaxis_title=x_axis,
        yaxis_title=target.capitalize()
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def predict_stroke_probability(features):
    # Scale the input features
    std_features = (features - X_train.mean(axis=0)) / X_train.std(axis=0)
    # Make predictions
    probability = model.predict_proba(std_features.reshape(1, -1))
    return probability[0][1]

def main():

    st.title("""
    Welcome to Our Stroke Prediction Application 🏥

    ## Project Introduction
    Welcome! This application aims to predict the likelihood of a brain stroke based on a variety of health and lifestyle factors. Our analysis is based on a dataset that includes key indicators related to stroke occurrences.

    """)

    st.write("### Data Overview", df.head())
    

    # Define target and features
    target = 'stroke'
    features = ['hypertension', 'heart_disease', 'ever_married', 'Residence_type',
                'avg_glucose_level', 'bmi', 'rounded_age', 'gender_Female',
                'gender_Male', 'gender_Other', 'smoking_status_Unknown',
                'smoking_status_formerly smoked', 'smoking_status_never smoked',
                'smoking_status_smokes', 'work_type_Govt_job', 'work_type_Never_worked',
                'work_type_Private', 'work_type_Self-employed', 'work_type_children']

    # Call the function to create the interactive scatter plot
    create_interactive_scatter_plot(df, target, features)

    
    st.write("### Data Features")
    # Input values for features
    hypertension = st.checkbox("Hypertension")
    heart_disease = st.checkbox("Heart Disease")
    ever_married = st.checkbox("Ever Married")
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level")
    bmi = st.number_input("BMI")
    rounded_age = st.number_input("Age")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    smoking_status = st.selectbox("Smoking Status", ["Unknown", "Formerly Smoked", "Never Smoked", "Smokes"])
    work_type = st.selectbox("Work Type", ["Govt Job", "Never Worked", "Private", "Self-employed", "Children"])

    # Map selected values to feature vector
    gender_mapping = {"Female": 1, "Male": 0, "Other": 2}
    gender_encoded = gender_mapping[gender]

    residence_mapping = {"Urban": 1, "Rural": 0}
    residence_encoded = residence_mapping[residence_type]

    smoking_mapping = {"Unknown": 0, "Formerly Smoked": 1, "Never Smoked": 2, "Smokes": 3}
    smoking_encoded = smoking_mapping[smoking_status]

    work_mapping = {"Govt Job": 0, "Never Worked": 1, "Private": 2, "Self-employed": 3, "Children": 4}
    work_encoded = work_mapping[work_type]

    # Prepare feature vector
    features = np.array([hypertension, heart_disease, ever_married, residence_encoded, avg_glucose_level, bmi, 
                         rounded_age, gender_encoded == 1, gender_encoded == 0, gender_encoded == 2, 
                         smoking_encoded == 0, smoking_encoded == 1, smoking_encoded == 2, smoking_encoded == 3, 
                         work_encoded == 0, work_encoded == 1, work_encoded == 2, work_encoded == 3, 
                         work_encoded == 4]).astype(int)

    # Predict probability
    if st.button("Predict"):
        probability = predict_stroke_probability(features)
        st.write(f"The probability of having a stroke is: {probability:.2f}")

if __name__ == "__main__":
    main()