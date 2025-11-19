import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="AutoDS Prediction App", layout="wide")

st.title("🎓 AutoDS ML Prediction Web App")

# -------------------------
# SAFE FILE LOADING
# -------------------------
def load_model():
    model_path = "Student_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None

    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_data():
    csv_path = "Employee_clean_Data.csv"
    if not os.path.exists(csv_path):
        st.error(f"❌ CSV file not found: {csv_path}")
        return None

    return pd.read_csv(csv_path)


model = load_model()
data = load_data()

if model is None or data is None:
    st.stop()

# -------------------------
# UI + PREDICTION
# -------------------------

st.subheader("📌 Data Preview")
st.dataframe(data.head())

st.subheader("📝 Enter Input Features")

input_features = {}
for col in data.columns:
    if data[col].dtype in ['int64', 'float64']:
        input_features[col] = st.number_input(
            f"{col}", value=float(data[col].mean())
        )
    else:
        input_features[col] = st.selectbox(
            f"{col}", options=list(data[col].unique())
        )

input_df = pd.DataFrame([input_features])

if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"📢 Prediction: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
