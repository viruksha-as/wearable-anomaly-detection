import streamlit as st
import joblib
import numpy as np

model = joblib.load("anomaly_model.pkl")

st.title("Wearable Anomaly Detection")

f1 = st.number_input("tBodyAcc-mean()-X", value=0.27)
f2 = st.number_input("tBodyAcc-mean()-Y", value=-0.02)
f3 = st.number_input("tBodyAcc-mean()-Z", value=-0.10)

if st.button("Predict"):
    data = np.array([[f1, f2, f3]])
    result = model.predict(data)

    if result[0] == -1:
        st.error("🚨 Anomaly Detected")
    else:
        st.success("✅ Normal Behavior")
