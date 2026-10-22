import pickle
import pandas as pd
import streamlit as st

st.title("Heart Disease Prediction")
st.write("Teaching demo: enter patient details and a trained Support Vector Classifier predicts whether heart disease is present.")

model = pickle.load(open("heart_trained_model.pkl", "rb"))

age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
thalach = st.number_input("Maximum heart rate achieved", min_value=60, max_value=220, value=150)

if st.button("Predict Heart Disease"):
    # Same encoding as the notebook: Female = 0, Male = 1
    sex_encoded = 1 if sex == "Male" else 0
    row = pd.DataFrame({"age": [age], "sex": [sex_encoded], "chol": [chol], "thalach": [thalach]})
    result = model.predict(row)[0]
    if result == 1:
        st.error("Prediction: HEART DISEASE likely")
    else:
        st.success("Prediction: NO heart disease likely")