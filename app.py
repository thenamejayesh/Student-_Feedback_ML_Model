import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="AutoDS Prediction App", layout="wide")
st.title("🎓 AutoDS ML Prediction Web App")

# -------------------------------------
# Load model safely
# -------------------------------------
def load_model():
    path = "Student_model.pkl"
    if not os.path.exists(path):
        st.error("❌ Model file NOT found.")
        return None
    
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()
if model is None:
    st.stop()

# -------------------------------------
# MANUALLY DEFINE FEATURES (IMPORTANT)
# -------------------------------------
TRAIN_FEATURES = ["age", "experience", "salary", "department"]

st.subheader("🔧 Enter Model Input Features")

# Create input boxes
user_input = {}

user_input["age"] = st.number_input("Age", min_value=18, max_value=70, value=25)
user_input["experience"] = st.number_input("Experience (Years)", min_value=0, max_value=40, value=1)
user_input["salary"] = st.number_input("Salary", min_value=10000, max_value=200000, value=30000)
user_input["department"] = st.selectbox("Department", ["IT", "HR", "Finance", "Sales", "Admin", "Other"])

# Convert to DataFrame
input_df = pd.DataFrame([user_input], columns=TRAIN_FEATURES)

if st.button("🔮 Predict"):
    try:
        pred = model.predict(input_df)
        st.success(f"📢 Prediction: **{pred[0]}**")
    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")
