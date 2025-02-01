import numpy as np
import pandas as pd
import joblib
import os
import streamlit as st

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")


@st.cache_data()
def load_df():
    return pd.read_csv("resources/df.csv", index_col="Unnamed: 0")


@st.cache_data()
def load_model():
    model = joblib.load("resources/model.pkl")
    return model


# load dataframe and model
df = load_df()
model = load_model()
features = ["failures_plus_higher_no", "avg_grade"]


st.title("Linear Regression to Predict UCI Student Performance")

# Goals
st.subheader("Goal")
st.write("""This dashboard provides an interactice exploration of a linear regression model 
         trained to predict student performance based on features such as grades, 
         failures, and the desire to pursue higher education.""")

# Approach
st.subheader("Approach")
st.image(os.path.join(os.getcwd(), "static", "Flowchart.png"))

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df)

# Selected Features
st.subheader("Selected Features")
st.markdown(r"""
| Feature Name               | Formula                     |
|----------------------------|-----------------------------|
| `avg_grade`                | $\frac{G_{1}\ +\ G_{2}}{2}$ |
| `failures_plus_higher_no`  | $failures + higher\_no$     |
""")

# Model Metrics
col1_metrics, col2_metrics = st.columns(2)
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["G3"], test_size=0.3, random_state=1
)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

with col1_metrics:
    st.subheader("Model Metrics")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

with col2_metrics:
    st.subheader("Model Parameters")
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    st.write(f"**Intercept:** {model.intercept_:.4f}")
    st.dataframe(coef_df)

# Key Insights
st.subheader("Key Insights")
st.markdown("<h6><u>Relative importance of features</u></h6>", unsafe_allow_html=True)
st.markdown(
    "- The 1<sup>st</sup> & 2<sup>nd</sup> period grades seem to be the main predictors of G<sub>3</sub>",
    unsafe_allow_html=True,
)
st.markdown(
    "- While not as significant, the number of failures and the desire to pursue higher education also contribute to the final predictions"
)


st.markdown("<h6><u>Conclusions</u></h6>", unsafe_allow_html=True)
st.markdown(
    "- Implement evidence-based learning strategies to better understand core concepts"
)
st.markdown("- Keep up with all assignments")
st.markdown("- Try your best to maintain high grades and minimize failures")
st.markdown("- Always have the desire to push harder and achieve big goals")

#Download Documentation
pdf_filename = "example.pdf"

# Streamlit download button for the PDF
st.download_button(
    label="Download Documentation",
    data=open("resources/Documentation.pdf", "rb").read(),
    file_name="resources/Documentation.pdf",
    mime="application/pdf"
)
