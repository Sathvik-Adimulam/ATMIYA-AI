import numpy as np
import joblib
import streamlit as st

st.set_page_config(layout="wide")


# cache & load model and encoders
@st.cache_data()
def load_resources():
    model = joblib.load("resources/model.pkl")
    higher_no_encoder = {"Yes": 0, "No": 1}
    return model, higher_no_encoder


# load model and preprocessors
model, higher_no_encoder = load_resources()

# Create playground
st.title("UCI Student Performance Dashboard")
with st.form(key="playground"):
    # init features
    features = np.zeros((1, 2))

    # grades
    G1 = st.select_slider("Select G1", options=list(range(21)))
    G2 = st.select_slider("Select G2", options=list(range(21)))
    if G1 and G2:
        features[0][1] = (G1 + G2) / 2

    # failures + higher_encoded
    failures = st.number_input(
        "Enter # of failures", min_value=0, value=0, step=1, format="%d"
    )
    higher = higher_no_encoder[st.radio("Select Higher Education", ["Yes", "No"])]

    if failures and higher:
        features[0][0] = failures + higher

    # submit
    submit_button = st.form_submit_button()

    if submit_button:
        pred = np.around(model.predict(features)).astype(int)[0]
        if pred > 20:
            pred = 20
        if pred < 0:
            pred = 0
        st.success(f"Predicted Grade: **{pred}**")
