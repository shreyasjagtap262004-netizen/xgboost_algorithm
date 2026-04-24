import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np

# 1. Load the model with error handling
@st.cache_resource
def load_model():
    try:
        # The model uses the gbtree booster with 100 trees [cite: 3, 14]
        with open('xgboost.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# 2. Page Configuration & Custom CSS
st.set_page_config(page_title="Student Success Predictor", layout="wide")

# FIXED: Changed 'unsafe_allow_value' to 'unsafe_allow_html'
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Student Performance Analytics")
st.markdown("Predictive modeling based on study habits, attendance, and background.")

# 3. Sidebar for User Inputs (Mapping the 9 required features)
st.sidebar.header("📋 Input Features")

def user_input_features():
    # Categorical / Binary Features [cite: 12, 13]
    gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    parent_education = st.sidebar.select_slider("Parent Education Level", options=[0, 1, 2, 3, 4], value=2)
    internet_access = st.sidebar.radio("Internet Access", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    extracurricular = st.sidebar.radio("Extracurricular Activities", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Numerical / Continuous Features [cite: 12, 13]
    age = st.sidebar.number_input("Age", min_value=12, max_value=25, value=18)
    study_hours = st.sidebar.slider("Study Hours Per Week", 0, 40, 15)
    # Attendance rate is stored as a float in the model [cite: 13]
    attendance = st.sidebar.slider("Attendance Rate (0.0 - 1.0)", 0.0, 1.0, 0.85)
    previous_score = st.sidebar.number_input("Previous Exam Score", 0, 100, 75)
    final_score = st.sidebar.number_input("Final Score Projection", 0, 100, 70)

    # Data structure matches model feature names exactly 
    data = {
        'gender': gender,
        'age': age,
        'study_hours_per_week': study_hours,
        'attendance_rate': attendance,
        'parent_education': parent_education,
        'internet_access': internet_access,
        'extracurricular': extracurricular,
        'previous_score': previous_score,
        'final_score': final_score
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Prediction Logic
if model is not None:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # SUCCESS_PROB FIX: Cast float32 to standard Python float for Streamlit
    # The model uses the binary:logistic objective [cite: 1, 349]
    success_prob = float(prediction_proba[0][1])

    # 5. Dashboard Display
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Student Profile Summary")
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    with col2:
        st.subheader("Success Prediction")
        st.metric(label="Probability of Success", value=f"{success_prob:.2%}")
        st.progress(success_prob)

        if prediction[0] == 1:
            st.success("### ✅ High Probability of Success")
        else:
            st.warning("### ⚠️ Support May Be Required")

else:
    st.error("Model file 'xgboost.pkl' not found.")

st.divider()
st.caption("Model Version: 1.0 | Objective: binary:logistic | Features: 9 [cite: 9, 349]")
