import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Employee Feedback Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
        .main-box { 
            background-color: #f0f9ff; 
            padding: 20px; 
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #6EE7B7, #3B82F6);
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
        .footer {
            text-align:center;
            padding:10px;
            margin-top:30px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Dashboard", "📈 Prediction"])

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    with open("Student_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------------- DASHBOARD PAGE ----------------------
if page == "🏠 Dashboard":
    st.title("📊 Employee Feedback Dashboard")

    st.markdown("<div class='main-box'>", unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload your employee CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("📄 Data Preview")
        st.dataframe(df)

        # Colored KPI metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'>Total Employees<br>{len(df)}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'>Avg Salary<br>{df['Salary'].mean():.2f}</div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'>Avg Experience<br>{df['Experience'].mean():.2f}</div>", unsafe_allow_html=True)

        st.subheader("📉 Salary Distribution")
        fig1 = px.histogram(df, x="Salary", nbins=30)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("🏢 Department Count")
        fig2 = px.pie(df, names="Department")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Upload a CSV file to view dashboard insights.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- PREDICTION PAGE ----------------------
elif page == "📈 Prediction":
    st.title("📈 Employee Feedback Prediction")

    st.markdown("<div class='main-box'>", unsafe_allow_html=True)

    st.write("Enter Employee Details for Prediction:")

    # Inputs
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    experience = st.number_input("Experience (Years)", min_value=0, max_value=40, value=2)
    salary = st.number_input("Salary", min_value=0, max_value=200000, value=50000)

    department = st.selectbox(
        "Department",
        options=[0, 1, 2],
        format_func=lambda x: {0: "IT", 1: "HR", 2: "Finance"}[x]
    )

    feature_order = ["Age", "Experience", "Salary", "Department"]

    input_data = pd.DataFrame([{
        "Age": age,
        "Experience": experience,
        "Salary": salary,
        "Department": department
    }])[feature_order]

    if st.button("🔍 Predict Feedback"):
        try:
            pred = model.predict(input_data)[0]
            st.success(f"✅ Predicted Feedback Score: **{pred}**")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- FOOTER ----------------------
st.markdown("<div class='footer'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
