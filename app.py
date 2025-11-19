 import streamlit as st
import pandas as pd
import pickle

st.title("Employee Feedback Prediction App")

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model():
    with open("Student_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.write("Enter employee details below to predict Feedback:")

# ---------------------- Input Fields ----------------------
age = st.number_input("Age", min_value=18, max_value=70, value=30)
experience = st.number_input("Experience (years)", min_value=0, max_value=40, value=2)
salary = st.number_input("Salary", min_value=0, max_value=200000, value=50000)

department = st.selectbox(
    "Department",
    options=[0, 1, 2],
    format_func=lambda x: {0: "IT", 1: "HR", 2: "Finance"}[x]
)

# ---------------------- Create Input DataFrame ----------------------
# MATCH EXACT TRAINING ORDER:
feature_order = ["Age", "Experience", "Salary", "Department"]

input_data = pd.DataFrame([{
    "Age": age,
    "Experience": experience,
    "Salary": salary,
    "Department": department
}])[feature_order]

# ---------------------- Prediction ----------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Feedback: {prediction}")
    except Exception as e:
        st.error(f"Error: {e}")
