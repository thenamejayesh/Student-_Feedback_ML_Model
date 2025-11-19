import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="AutoDS ML Prediction", layout="wide")
st.title("🎓 AutoDS ML Prediction App")

# ----------------------------------------------------
# Load Model File
# ----------------------------------------------------
def load_model():
    model_path = "Student_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()
if model is None:
    st.stop()

# ----------------------------------------------------
# FEATURES USED IN MODEL TRAINING
# ----------------------------------------------------
FEATURES = ["Age", "Experience", "Salary", "Department"]

st.subheader("📝 Enter Feature Values")

# ----------------------------------------------------
# User Input Section
# ----------------------------------------------------
age = st.number_input("Age", min_value=18, max_value=70, value=25)
experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=1)
salary = st.number_input("Salary", min_value=10000, max_value=300000, value=30000)
department = st.selectbox("Department", ["IT", "HR", "Finance", "Sales", "Admin", "Other"])

# Create DataFrame for prediction
input_data = pd.DataFrame([{
    "Age": Age,
    "Experience": Experience,
    "Salary": Salary,
    "Department": Department
}])

# ----------------------------------------------------
# Prediction Button
# ----------------------------------------------------
if st.button("🔮 Predict"):
    try:
        pred = model.predict(input_data)
        st.success(f"📢 Prediction Result: **{pred[0]}**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
