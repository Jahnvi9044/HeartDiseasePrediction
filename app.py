import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the mapping for categorical variables
sex_mapping = {'F': 0, 'M': 1}
chest_pain_mapping = {'ASY': 0, 'NAP': 1, 'ATA': 2, 'TA': 3}
resting_ecg_mapping = {'Normal': 0, 'LVH': 1, 'ST': 2}
exercise_angina_mapping = {'N': 0, 'Y': 1}
st_slope_mapping = {'Up': 0, 'Flat': 1, 'Down': 2}

# Define a function to make predictions
def predict_heart_disease(input_data):
    # Convert categorical data using the mappings
    input_data['Sex'] = sex_mapping[input_data['Sex']]
    input_data['ChestPainType'] = chest_pain_mapping[input_data['ChestPainType']]
    input_data['RestingECG'] = resting_ecg_mapping[input_data['RestingECG']]
    input_data['ExerciseAngina'] = exercise_angina_mapping[input_data['ExerciseAngina']]
    input_data['ST_Slope'] = st_slope_mapping[input_data['ST_Slope']]
    columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
           'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
           'Oldpeak', 'ST_Slope']
    input_df = pd.DataFrame([input_data])
    input_df = input_df[columns]  # Ensure columns are in the same order
    prediction = model.predict(input_df)
    return prediction[0]


# Streamlit app layout
st.title("Heart Disease Prediction")

# Collect user input
st.header("Enter Patient Data:")
age = st.number_input("Age", min_value=1, max_value=120, value=55)
sex = st.selectbox("Sex", options=sex_mapping.keys())
chest_pain_type = st.selectbox("Chest Pain Type", options=chest_pain_mapping.keys())
resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=130)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=250)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
resting_ecg = st.selectbox("Resting ECG", options=resting_ecg_mapping.keys())
max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=160)
exercise_angina = st.selectbox("Exercise-Induced Angina", options=exercise_angina_mapping.keys())
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.2)
st_slope = st.selectbox("ST Slope", options=st_slope_mapping.keys())

# Create a dictionary for the input data
input_data = {
    'Age': age,
    'Sex': sex,
    'ChestPainType': chest_pain_type,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'RestingECG': resting_ecg,
    'MaxHR': max_hr,
    'ExerciseAngina': exercise_angina,
    'Oldpeak': oldpeak,
    'ST_Slope': st_slope,
}

# Predict and display the result
if st.button("Predict"):
    prediction = predict_heart_disease(input_data)
    if prediction == 1:
        st.write("The model predicts: **Heart Disease Detected**")
    else:
        st.write("The model predicts: **No Heart Disease Detected**")

# Additional info or instructions
st.write("Note: This prediction is based on a machine learning model and should not be considered as medical advice.")

