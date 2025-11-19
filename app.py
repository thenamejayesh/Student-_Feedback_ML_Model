import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Employee ML Prediction App", layout="wide")
st.title("🏢 Employee ML Prediction App (AutoDS)")

# --------------------------------------------------------
# Load Pickle Model Safely
# --------------------------------------------------------
def load_model():
    model_path = "Student_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Student_model.pkl file not found in the deployed folder!")
        return None
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# --------------------------------------------------------
# MANUAL FEATURE LIST (You provided)
# --------------------------------------------------------
FEATURES = ["age", "experience", "salary", "department"]

st.subheader("📝 Enter Employee Details")

# --------------------------------------------------------
# Input UI
# --------------------------------------------------------
age = st.number_input("Age", min_value=18, max_value=70, value=25)
experience = st
