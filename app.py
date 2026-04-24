import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np

# 1. Load the model with error handling
@st.cache_resource
def load_model():
    try:
        with open('xgboost.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# 2. Page Configuration & Custom CSS for a "Dynamic" feel
st.set_page_config(page_title="Student Success Predictor", layout="wide")
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
    """, unsafe_allow_value=True)

st.title("🎓 Student Performance Analytics")
st.markdown("Predictive modeling based on study habits, attendance, and background.")

# 3. Sidebar for User Inputs (Defining the 9 required features)
st.sidebar.header("📋 Input Features")

def user_input_features():
    # Categorical / Binary Features
    gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    parent_education = st.sidebar.select_slider("Parent Education Level", options=[0, 1, 2, 3, 4], value=2)
    internet_access = st.sidebar.segmented_control("Internet Access", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    extracurricular = st.sidebar.radio("Extracurricular Activities", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Numerical / Continuous Features
    age = st.sidebar.number_input("Age", min_value=12, max_value=25, value=18)
    study_hours = st.sidebar.slider("Study Hours Per Week", 0, 40, 15)
    attendance = st.sidebar.slider("Attendance Rate (0.0 - 1.0)", 0.0, 1.0, 0.85)
    previous_score = st.sidebar.number_input("Previous Exam Score", 0, 100, 75)
    final_score = st.sidebar.number_input("Final Score Projection", 0, 100, 70)

    # Data structure MUST match model feature names
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
    # Perform prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # --- THE FIX: Cast float32 to standard Python float ---
    # Your model uses binary:logistic with float32 precision
    success_prob = float(prediction_proba[0][1])

    # 5. Dynamic
