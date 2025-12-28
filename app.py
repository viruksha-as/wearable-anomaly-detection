import streamlit as st
import joblib
import numpy as np

# load model and features
model = joblib.load("anomaly_model.pkl")
features = joblib.load("features.pkl")

st.title("Wearable Anomaly Detection")

f1 = st.number_input("tBodyAcc-mean()-X", value=0.27)
f2 = st.number_input("tBodyAcc-mean()-Y", value=-0.02)
f3 = st.number_input("tBodyAcc-mean()-Z", value=-0.10)

if st.button("Predict"):
    input_data = np.zeros(len(features))

    input_data[features.index("tBodyAcc-mean()-X")] = f1
    input_data[features.index("tBodyAcc-mean()-Y")] = f2
    input_data[features.index("tBodyAcc-mean()-Z")] = f3

    result = model.predict([input_data])

    if result[0] == -1:
        st.error("🚨 Anomaly Detected")
    else:
        st.success("✅ Normal Behavior")
