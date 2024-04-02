import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Load the CSV files to df
df = pd.read_csv('data\cleaned\dataset_encoded.csv')
df_2 = pd.read_csv('data\cleaned\stroke_data_cleaned.csv')
df = df.dropna()

# Load the trained model and associated data
with open("notebooks\Logistic_Regression.pkl", "rb") as f:
    trained_data = pickle.load(f)

# Extract the model and other data
model = trained_data['model']
scaler = trained_data['scaler']  # Assuming your saved trained data includes the scaler
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


def create_interactive_histogram(df):
    st.write('## Interactive Histogram Plot')

    # Sidebar for user input:
    column_options = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                      'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
                      'smoking_status', 'stroke', 'rounded_age']
    selected_column = st.sidebar.selectbox('Select the column for the histogram plot:', column_options)

    # Create a Plotly Express histogram plot
    fig = px.histogram(df, x=selected_column, color='stroke', nbins=20, marginal='rug')
    
    # Update layout if desired
    fig.update_layout(
        title=f'Histogram Plot of {selected_column.capitalize()}',
        xaxis_title=selected_column.capitalize(),
        yaxis_title='Count'
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
def predict_stroke_probability(features):
    # Check for NaN values
    if np.isnan(features).any():
        raise ValueError("One or more input features are missing. Please fill all the fields.")

    # Scale the input features
    std_features = (features - X_train.mean(axis=0)) / X_train.std(axis=0)
    
    # Make predictions
    probability = model.predict_proba(std_features.reshape(1, -1))
    return probability[0][1]


def main():
    gender_mapping = {"Female": 1, "Male": 0, "Other": 2}
    residence_mapping = {"Urban": 1, "Rural": 0}
    smoking_mapping = {"Unknown": 0, "Formerly Smoked": 1, "Never Smoked": 2, "Smokes": 3}
    work_mapping = {"Govt Job": 0, "Never Worked": 1, "Private": 2, "Self-employed": 3, "Children": 4}
    

    st.title("""
    Welcome to Our Stroke Prediction Application 🏥

    ## Project Introduction
    Welcome! This application aims to predict the likelihood of a brain stroke based on a variety of health and lifestyle factors. Our analysis is based on a dataset that includes key indicators related to stroke occurrences.

    """)
    st.write('## Data Overview')
    st.write('', df.head(5200))

    # Call the function to create the interactive plots

    # create_interactive_scatter_plot(df_2, target, features)
    create_interactive_histogram(df_2)

    st.write('##  Brain Stroke Prediction')
    st.write("### Data Features")
    st.write("Please select the suitable data features according to our patient.")
    user_inputs = {}

    # Input values for features, ensure all inputs are required by checking their values before prediction
    user_inputs['hypertension'] = st.checkbox("Hypertension")
    user_inputs['heart_disease'] = st.checkbox("Heart Disease")
    user_inputs['ever_married'] = st.checkbox("Ever Married")
    user_inputs['residence_type'] = st.selectbox("Residence Type", ["Urban", "Rural"])
    user_inputs['avg_glucose_level'] = st.number_input("Average Glucose Level", min_value=50.0, step=5.0, format="%.2f")
    user_inputs['bmi'] = st.number_input("BMI", min_value=18.5, step=1.5, format="%.2f")
    user_inputs['age'] = st.number_input("Age", min_value=0, step=5, format="%d")
    user_inputs['gender'] = st.selectbox("Gender", ["Female", "Male"])
    user_inputs['smoking_status'] = st.selectbox("Smoking Status", ["Unknown", "Formerly Smoked", "Never Smoked", "Smokes"])
    user_inputs['work_type'] = st.selectbox("Work Type", ["Govt Job", "Never Worked", "Private", "Self-employed", "Children"])
    
    if st.button("Predict"):
        # Initialize an empty list for the final features
        final_features = []

        # Add numerical features directly
        final_features.extend([
            user_inputs['hypertension'],
            user_inputs['heart_disease'],
            1 if user_inputs['ever_married'] else 0,
            user_inputs['avg_glucose_level'],
            user_inputs['bmi'],
            user_inputs['age']
        ])

        # One-hot encode 'residence_type'
        final_features.extend([
            1 if user_inputs['residence_type'] == "Urban" else 0,
            1 if user_inputs['residence_type'] == "Rural" else 0,
        ])

        # One-hot encode 'gender'
        final_features.extend([
            1 if user_inputs['gender'] == "Female" else 0,
            1 if user_inputs['gender'] == "Male" else 0,
        ])

        # One-hot encode 'smoking_status'
        smoking_status_categories = ["Unknown", "Formerly Smoked", "Never Smoked", "Smokes"]
        final_features.extend([1 if user_inputs['smoking_status'] == category else 0 for category in smoking_status_categories])

        # One-hot encode 'work_type'
        work_type_categories = ["Govt Job", "Never Worked", "Private", "Self-employed", "Children"]
        final_features.extend([1 if user_inputs['work_type'] == category else 0 for category in work_type_categories])

        # Convert to NumPy array and reshape for a single sample
        features_array = np.array(final_features).reshape(1, -1)
        
        # Verify the final array size matches the scaler's expectation
        if features_array.shape[1] != 19:
            st.error("Unexpected number of features. Please check the input data.")
            return

        # Scale the features using the loaded scaler
        features_scaled = scaler.transform(features_array)

        # Make prediction
        probability = model.predict_proba(features_scaled)
        percentage = round(probability[0][1] * 100, 2)
        st.write(f"The probability of having a brain stroke is: {percentage}%")



if __name__ == "__main__":
    main()