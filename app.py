 import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# --- CONFIG: update these paths if your files are named differently ---
MODEL_PATH = "/mnt/data/Student_model (1).pkl"
DATA_CSV_PATH = "/mnt/data/student_scores (1).csv"
# -----------------------------------------------------------------------

st.set_page_config(page_title="Streamlit Prediction App", layout="centered")
st.title("Prediction App — Enter all features and press Predict")
st.markdown(
    "This app loads a pickled model and the training CSV to build a label-encoder "
    "for categorical columns (department). All inputs are required before prediction."
)

@st.cache_data(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

@st.cache_data(show_spinner=False)
def load_train_csv(path):
    if not os.path.exists(path):
        return None, f"CSV file not found at: {path}"
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, f"Error loading CSV: {e}"

model, model_err = load_model(MODEL_PATH)
train_df, csv_err = load_train_csv(DATA_CSV_PATH)

if model_err:
    st.error(model_err)
if csv_err:
    st.error(csv_err)

if model is None or train_df is None:
    st.info("Fix paths at top of app.py if your files are located elsewhere.")
    st.stop()

expected_features = ["age", "experience", "salary", "department"]
missing = [c for c in expected_features if c not in train_df.columns]
if missing:
    st.warning(f"Warning: training CSV doesn't contain expected columns: {missing}. "
               "Label encoding will try to proceed if 'department' exists.")

if "department" in train_df.columns:
    dept_values = train_df["department"].dropna().astype(str).unique().tolist()
    le = LabelEncoder()
    le.fit(dept_values)
    dept_options = list(le.classes_)
else:
    le = None
    dept_options = []

st.subheader("Input features")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=200,
            value=int(train_df["age"].median()) if "age" in train_df.columns else 25
        )
        experience = st.number_input(
            "Experience (years)",
            min_value=0.0,
            max_value=100.0,
            value=float(train_df["experience"].median()) if "experience" in train_df.columns else 1.0,
            step=0.5
        )
    with col2:
        salary = st.number_input(
            "Salary",
            min_value=0.0,
            value=float(train_df["salary"].median()) if "salary" in train_df.columns else 30000.0,
            step=100.0,
            format="%.2f"
        )
        if dept_options:
            department = st.selectbox("Department", options=dept_options)
        else:
            department = st.text_input("Department (no department list available in CSV)", value="")
    st.write("")  # spacer
    submit = st.form_submit_button("Predict")

if not submit:
    st.info("Enter all features and click **Predict**.")
    st.stop()

if any(v is None or (isinstance(v, str) and v.strip() == "") for v in [age, experience, salary, department]):
    st.error("All features are required. Please fill every field.")
    st.stop()

try:
    if le is not None:
        if str(department) in le.classes_:
            dept_enc = le.transform([str(department)])[0]
        else:
            dept_enc = -1
    else:
        dept_enc = str(department)

    input_array = np.array([[age, experience, salary, dept_enc]], dtype=object)

    try:
        input_array = input_array.astype(float)
    except Exception:
        pass

    prediction = model.predict(input_array)

    st.subheader("Prediction result")
    try:
        result = prediction[0] if hasattr(prediction, "__len__") else prediction
        st.success(f"Predicted value: **{result}**")
    except Exception:
        st.success(f"Predicted value: **{prediction}**")

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(input_array)
            class_labels = getattr(model, "classes_", None)
            if class_labels is not None:
                proba_df = pd.DataFrame(proba, columns=[str(c) for c in class_labels])
            else:
                proba_df = pd.DataFrame(proba, columns=[f"prob_{i}" for i in range(proba.shape[1])])
            st.subheader("Prediction probabilities")
            st.dataframe(proba_df)
        except Exception as e:
            st.info(f"Model has predict_proba but failed to produce probabilities: {e}")

    if hasattr(model, "score"):
        st.info("Model exposes a `score` method (we don't compute it here because a true label is required).")

except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

st.markdown("---")
st.caption(
    "If you get `could not convert string to float` errors, ensure the model was trained with label-encoded "
    "`department`. This app builds the encoder from the provided CSV and applies the same mapping."
)
