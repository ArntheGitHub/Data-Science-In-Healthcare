# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# # Load the CSV files to df
# df = pd.read_csv('data\cleaned\dataset_encoded.csv')

# # Load the trained model and associated data
# with open("notebooks\Logistic_Regression.pkl", "rb") as f:
#     trained_data = pickle.load(f)

# # Extract the model and other data
# # model = trained_data['model']
# # features = trained_data['features'] 


# def classify(num):
#     pass

# def main():

#     st.title("""
#     # Welcome to Our Stroke Prediction Application 🏥

#     ## Project Introduction
#     Welcome! This application aims to predict the likelihood of a brain stroke based on a variety of health and lifestyle factors. Our analysis is based on a dataset that includes key indicators related to stroke occurrences.

#     """)

#     st.write("### Data Overview", df.head())

# if __name__ == "__main__":
#     main()
    
# # # Define function to preprocess input features
# # def preprocess_features(hypertension, heart_disease, ever_married, residence_type, avg_glucose_level, bmi, rounded_age, gender, smoking_status, work_type):
# #     # Map selected values to feature vector
# #     gender_mapping = {"Female": 1, "Male": 0, "Other": 2}
# #     gender_encoded = gender_mapping[gender]

# #     residence_mapping = {"Urban": 1, "Rural": 0}
# #     residence_encoded = residence_mapping[residence_type]

# #     smoking_mapping = {"Unknown": 0, "Formerly Smoked": 1, "Never Smoked": 2, "Smokes": 3}
# #     smoking_encoded = smoking_mapping[smoking_status]

# #     work_mapping = {"Govt Job": 0, "Never Worked": 1, "Private": 2, "Self-employed": 3, "Children": 4}
# #     work_encoded = work_mapping[work_type]

# #     # Prepare feature vector
# #     processed_features = np.array([hypertension, heart_disease, ever_married, residence_encoded, avg_glucose_level, bmi, 
# #                                     rounded_age, gender_encoded == 1, gender_encoded == 0, gender_encoded == 2, 
# #                                     smoking_encoded == 0, smoking_encoded == 1, smoking_encoded == 2, smoking_encoded == 3, 
# #                                     work_encoded == 0, work_encoded == 1, work_encoded == 2, work_encoded == 3, 
# #                                     work_encoded == 4]).astype(int)

# #     return processed_features

# # # Function to predict stroke probability
# # def predict_stroke_probability(processed_features):
# #     # Make predictions
# #     probability = model.predict_proba(processed_features.reshape(1, -1))
# #     return probability[0][1]

# # # Streamlit app
# # def main():
# #     st.title("Stroke Prediction App")

# #     # Input values for features
# #     hypertension = st.checkbox("Hypertension")
# #     heart_disease = st.checkbox("Heart Disease")
# #     ever_married = st.checkbox("Ever Married")
# #     residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
# #     avg_glucose_level = st.number_input("Average Glucose Level")
# #     bmi = st.number_input("BMI")
# #     rounded_age = st.number_input("Age")
# #     gender = st.selectbox("Gender", ["Female", "Male", "Other"])
# #     smoking_status = st.selectbox("Smoking Status", ["Unknown", "Formerly Smoked", "Never Smoked", "Smokes"])
# #     work_type = st.selectbox("Work Type", ["Govt Job", "Never Worked", "Private", "Self-employed", "Children"])

# #     # Predict probability
# #     if st.button("Predict"):
# #         # Preprocess input features
# #         processed_features = preprocess_features(hypertension, heart_disease, ever_married, residence_type, avg_glucose_level, bmi, rounded_age, gender, smoking_status, work_type)
        
# #         # Predict stroke probability
# #         probability = predict_stroke_probability(processed_features)
# #         st.write(f"The probability of having a stroke is: {probability:.2f}")


import streamlit as st

def main():
    st.title("Título de mi aplicación")

if __name__ == "__main__":
    main()