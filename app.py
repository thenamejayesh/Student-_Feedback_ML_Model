import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "Student_model (1).pkl"
DATA_CSV_PATH = "student_scores (1).csv"

st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("Student Score Prediction App")
st.write("Enter all features and click Predict.")

@st.cache_data
def load_model(path):
    if not os.path.exists(path):
        return None, f"Model not found: {path}"
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, f"Model loading error: {e}"

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None, f"CSV not found: {path}"
    try:
        return pd.read_csv(path), None
    except Exception as e:
        return None, f"CSV loading error: {e}"

model, model_err = load_model(MODEL_PATH)
train_df, csv_err = load_csv(DATA_CSV_PATH)

if model_err:
    st.error(model_err)
if csv_err:
    st.error(csv_err)

if model is None or train_df is None:
    st.stop()

expected_cols = ["age", "experience", "salary", "department"]
missing = [c for c in expected_cols if c not in train_df.columns]
if missing:
    st.warning(f"CSV missing columns: {missing}")

# Label encoding for department
if "department" in train_df.columns:
    le = LabelEnc
