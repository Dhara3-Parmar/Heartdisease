
import streamlit as st
import joblib
import streamlit.components.v1 as components

from input import validate_input, predict_result

st.set_page_config(
    page_title="HeartCare",
    layout="centered"
)

model = joblib.load("heart_model_v20261013_160903.joblib")

feature_names = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

st.markdown("""
<style>
.stApp {
    background: #f3f6fb;
}

header, footer {
    visibility: hidden;
}

.main .block-container {
    max-width: 850px;
    padding: 25px 20px;
}

.header {
    background: #142b53;
    color: white;
    padding: 25px;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 25px;
}

.section {
    background: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 18px;
    border: 1px solid #e2e8f0;
}

.section h3 {
    color: #142b53;
}

.stButton button {
    background: #142b53;
    color: white;
    width: 100%;
    height: 45px;
    border-radius: 8px;
}

.stButton button:hover {
    background: #244577;
    color: white;
}

@media(max-width: 600px) {
    .main .block-container {
        padding: 12px 10px;
    }

    .header {
        padding: 18px;
    }

    .header h1 {
        font-size: 24px;
    }

    .section {
        padding: 14px;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>HeartCare</h1>
    <p>Heart Disease Risk Prediction System</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section">
    <h3>Patient Information</h3>
    <p>Enter patient medical details below.</p>
</div>
""", unsafe_allow_html=True)

age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 0, 300, 120)
chol = st.number_input("Cholesterol", 0, 700, 200)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", 0, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict Heart Disease Risk"):

    values = [
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ]

    try:
        input_data = validate_input(values, feature_names)

        prediction, probabilities = predict_result(
            model, input_data
        )

        confidence = max(probabilities) * 100

        if prediction == 1:
            st.error("Higher Risk Prediction")
        else:
            st.success("Lower Risk Prediction")

        st.write("Model Confidence:", round(confidence, 2), "%")

    except Exception as e:
        st.error(str(e))

st.markdown("""
<div style="text-align:center;color:#64748b;padding:20px;">
    HeartCare | Educational Purpose Only
    <br>
    Not a substitute for professional medical diagnosis.
</div>
""", unsafe_allow_html=True)
