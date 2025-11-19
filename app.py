import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "Student_model (1).pkl"
DATA_PATH = "student_scores (1).csv"

st.title("Prediction App (Label Encoded)")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

# Label Encode department
le = LabelEncoder()
le.fit(df["department"].astype(str))

st.subheader("Enter Input Values")

with st.form("input_form"):
    age = st.number_input("Age", min_value=0, max_value=100)
    experience = st.number_input("Experience", min_value=0.0, step=0.5)
    salary = st.number_input("Salary", min_value=0.0, step=100.0)
    department = st.selectbox("Department", le.classes_)

    submit = st.form_submit_button("Predict")

if submit:
    dept_enc = le.transform([department])[0]
    X = np.array([[age, experience, salary, dept_enc]])

    pred = model.predict(X)[0]
    st.success(f"Prediction: {pred}")
