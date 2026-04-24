import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np

# 1. Load the model
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

st.set_page_config(page_title="Student Success Predictor", layout="wide")

st.title("🎓 Student Performance Prediction")
st.info("💡 Note: If the probability stays at 99%, try lowering the 'Final Score Projection' and 'Previous Score' values.")

# 2. Sidebar Inputs
st.sidebar.header("Student Parameters")

def user_input_features():
    # Use standard scales (0-100) often found in student datasets
    gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    age = st.sidebar.number_input("Age", 10, 30, 18)
    study_hours = st.sidebar.slider("Study Hours Per Week", 0, 50, 15)
    # Ensure attendance is a float if model expects 0.0-1.0
    attendance = st.sidebar.slider("Attendance Rate (0.0 to 1.0)", 0.0, 1.0, 0.85)
    parent_ed = st.sidebar.slider("Parent Education (0-4)", 0, 4, 2)
    internet = st.sidebar.radio("Internet Access", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    extra = st.sidebar.radio("Extracurricular", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # These two are likely the dominant features causing the 99% stuck result
    prev_score = st.sidebar.slider("Previous Score", 0, 100, 75)
    final_score = st.sidebar.slider("Final Score Projection", 0, 100, 70)

    # Dictionary keys MUST match the model's feature_names exactly
    data = {
        'gender': int(gender),
        'age': int(age),
        'study_hours_per_week': int(study_hours),
        'attendance_rate': float(attendance),
        'parent_education': int(parent_ed),
        'internet_access': int(internet),
        'extracurricular': int(extra),
        'previous_score': int(prev_score),
        'final_score': int(final_score)
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# 3. Main Dashboard
if model:
    # CRITICAL FIX: Ensure the column order matches the model exactly
    # Your model expects: gender, age, study_hours_per_week, attendance_rate, 
    # parent_education, internet_access, extracurricular, previous_score, final_score
    expected_order = [
        'gender', 'age', 'study_hours_per_week', 'attendance_rate', 
        'parent_education', 'internet_access', 'extracurricular', 
        'previous_score', 'final_score'
    ]
    df = df[expected_order]

    # Prediction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    # Cast float32 to float for streamlit progress bar
    success_prob = float(prediction_proba[0][1])

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Input Data")
        st.write(df)
        
    with col2:
        st.subheader("Result")
        st.metric("Success Probability", f"{success_prob:.2%}")
        st.progress(success_prob)
        
        if success_prob > 0.5:
            st.success("The student is likely to pass.")
        else:
            st.error("The student is at risk of failing.")

    # Debug Section (to see why it's stuck)
    with st.expander("Why is the prediction always the same?"):
        st.write("""
        1. **Feature Leakage**: Your model includes `final_score` as an input. If the model defines 'Success' as having a score over 50, then any `final_score` you enter above 50 will automatically result in 99% probability.
        2. **Dominant Features**: Try setting `final_score` and `previous_score` to very low values (like 10). If the probability drops, it means those features are overshadowing the others.
        3. **Scale**: If your model was trained on percentages (0-100) but you give it a rate (0.85), it interprets attendance as 0.85%.
        """)
