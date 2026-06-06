import streamlit as st
import numpy as np
import pickle

# 1. Load the saved model and scaler
# Make sure these match the filenames exported by train_model.py
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# 2. Frontend Title and Description
st.title("Diabetes Prediction System")
st.write("Enter the medical details to check the diabetic status.")

# 3. User Input Fields (Arranged in columns for a cleaner UI)
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure value", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness value", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI value", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function value", min_value=0.000, max_value=3.000, value=0.5)
    age = st.number_input("Age of the Person", min_value=0, max_value=120, value=30)

# 4. Prediction Logic
if st.button("Diabetes Test Result"):
    
    # Store user inputs as a tuple
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    
    # Convert to numpy array and reshape for a single instance
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Standardize the user input using the loaded scaler
    std_data = scaler.transform(input_data_reshaped)
    
    # Make the prediction
    prediction = loaded_model.predict(std_data)
    
    # Display the result
    if prediction[0] == 0:
        st.success("The person is NOT diabetic.")
    else:
        st.error("The person is diabetic.")