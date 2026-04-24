import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# 1. Load the model
@st.cache_resource
def load_model():
    with open('xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# 2. Page Configuration
st.set_page_config(page_title="Student Success Predictor", layout="wide")
st.title("🎓 Student Performance Prediction")
st.markdown("Adjust the student attributes below to predict the likelihood of success.")

# 3. Dynamic Sidebar Inputs
st.sidebar.header("Student Metrics")

def user_input_features():
    # Categorical Inputs (based on the int types in your model)
    gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    parent_education = st.sidebar.slider("Parent Education Level (0-4)", 0, 4, 2)
    internet_access = st.sidebar.radio("Internet Access", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    extracurricular = st.sidebar.radio("Extracurricular Activities", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Numerical Inputs
    age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=18)
    study_hours = st.sidebar.slider("Study Hours Per Week", 0, 168, 20)
    attendance = st.sidebar.slider("Attendance Rate (%)", 0.0, 1.0, 0.85)
    previous_score = st.sidebar.number_input("Previous Score", 0, 100, 75)
    final_score_input = st.sidebar.number_input("Current/Final Score Metric", 0, 100, 70)

    # Dataframe construction matches your model's feature names 
    data = {
        'gender': gender,
        'age': age,
        'study_hours_per_week': study_hours,
        'attendance_rate': attendance,
        'parent_education': parent_education,
        'internet_access': internet_access,
        'extracurricular': extracurricular,
        'previous_score': previous_score,
        'final_score': final_score_input
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# 4. Main Display Area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Selected Parameters")
    st.write(df.T.rename(columns={0: 'Value'}))

with col2:
    st.subheader("Prediction Result")
    
    # Run prediction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    # Visualizing Probability
    success_prob = prediction_proba[0][1]
    st.metric(label="Likelihood of Success", value=f"{success_prob:.2%}")
    st.progress(success_prob)

    if prediction[0] == 1:
        st.success("Result: **High Probability of Success**")
    else:
        st.warning("Result: **Lower Probability of Success**")

st.info("Note: This model was trained on 9 specific student features including attendance and study habits.")
