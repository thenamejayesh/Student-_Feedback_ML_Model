import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Load Model & Data
# -------------------------
@st.cache_resource
def load_model():
    with open("Student_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

@st.cache_data
def load_data():
    return pd.read_csv("Employee_clean_Data.csv")

model = load_model()
data = load_data()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AutoDS Prediction App", layout="wide")

st.title("🎓 AutoDS ML Prediction Web App")
st.write("Upload input values to generate predictions using the trained model.")

st.subheader("📌 Data Preview")
st.dataframe(data.head())

st.subheader("📝 Enter Input Features")

# AUTO CREATE INPUT BOXES BASED ON DATA COLUMNS
input_features = {}
for col in data.columns:
    # Exclude target column if present
    if data[col].dtype in ['int64', 'float64']:
        input_features[col] = st.number_input(f"Enter {col}", value=float(data[col].mean()))
    else:
        input_features[col] = st.selectbox(f"Select {col}", options=data[col].unique())

# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_features])

if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"📢 Model Prediction: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.write("Built with ❤️ using Streamlit")
