import streamlit as st
import pandas as pd
import joblib
import pickle

# Page config
st.set_page_config(page_title="Income Predictor", layout="centered")

# Title
st.title("💼 Adult Income Prediction App")
st.markdown("### Predict whether a person earns more than 50K")

# Load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

# Inputs
st.write("### Enter details below:")

age = st.slider("Age", 18, 90)
education_num = st.slider("Education Number", 1, 16)
hours_per_week = st.slider("Hours per week", 1, 100)
workclass = st.selectbox("Workclass", ["Private", "Self-emp", "Government"])
sex = st.selectbox("Gender", ["Male", "Female"])

# Predict button
if st.button("🔍 Predict Income"):

    # Show input
    st.write("### Your Input:")
    st.write({
        "Age": age,
        "Education": education_num,
        "Hours per week": hours_per_week,
        "Workclass": workclass,
        "Gender": sex
    })

    # Create input dataframe
    input_data = pd.DataFrame([[age, education_num, hours_per_week]],
                              columns=['age','education.num','hours.per.week'])

    # Add missing columns
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[columns]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.success("✅ High Income (>50K)")
    else:
       st.warning("⚠️ Low Income (<=50K)")


# Footer
st.markdown("---")
st.markdown("Created by Saniya | ML Project")
