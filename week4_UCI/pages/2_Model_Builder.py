import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")


@st.cache_data()
def load_df():
    return pd.read_csv("df.csv", index_col="Unnamed: 0")


df = load_df()

# Streamlit App
st.title("Feature Testing")

# Sidebar for selecting features
st.header("Feature Selection")
features = st.multiselect(
    "Select features for training:",
    options=df.drop(columns=["G3"]).columns,  # Exclude the target column from options
    default=["avg_grade", "failures_plus_higher_no"],
)

col1, col2 = st.columns(2)
# Filter DataFrame to display selected columns
if features:
    columns_to_display = features + ["G3"]
else:
    columns_to_display = ["G3"]

filtered_df = df[columns_to_display]

# Display the filtered dataframe
with col1:
    st.subheader("Filtered DataFrame")
    st.dataframe(filtered_df)

    train_button = st.button("Train")

# Train model if features are selected
if train_button:
    # Prepare the data
    if not features:
        st.warning("Please select at least one feature to train the mod")
    X = df[features]
    y = df["G3"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions and calculate metrics
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # Display metrics
    with col2:
        st.subheader("Model Metrics")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # Display coefficients and intercept
        st.subheader("Model Parameters")
        coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
        st.write(f"**Intercept:** {model.intercept_:.4f}")
        st.dataframe(coef_df)
