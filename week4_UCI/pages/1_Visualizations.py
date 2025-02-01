import streamlit as st
import pandas as pd

import plotly.express as px


st.set_page_config(layout="wide")


@st.cache_data()
def load_df():
    return pd.read_csv("resources/df.csv", index_col="Unnamed: 0")


df = load_df()
st.title("Visualizations")
tab1, tab2 = st.tabs(["Graphs", "Correlation Matrix"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        # graph hist of avg_grade
        avg_grade_hist = px.histogram(
            df["avg_grade"], title="Histogram of Average Grades"
        )
        avg_grade_hist.update_layout(xaxis_title="Average Grade")
        st.plotly_chart(avg_grade_hist)

        # graph violin plot of avg_grade
        avg_grade_violin = px.violin(
            df["avg_grade"], title="Violin Plot of Average Grades"
        )
        avg_grade_violin.update_layout(xaxis_title="", yaxis_title="Average Grade")
        st.plotly_chart(avg_grade_violin)

    with col2:
        # graph failures
        failures_fig = px.bar(
            df["failures"].value_counts(), title="Bar Graph of Failures"
        )
        st.plotly_chart(failures_fig)

        # graph higher_yes
        higher_fig = px.bar(
            df["higher_yes"].value_counts(), title="Bar Graph of Higher Education"
        )
        higher_fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 1],  # Values in the data
                ticktext=["No", "Yes"],  # Text to display
            ),
            xaxis_title="Higher Education",
        )
        st.plotly_chart(higher_fig)

# display correlation matrix
with tab2:
    corr_matrix = (
        df.select_dtypes(exclude=["object"])
        .corr()["G3"]
        .sort_values(ascending=False)
        .iloc[1:]
        .round(3)
        .to_frame()
    )
    fig = px.imshow(
        corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r"
    )
    fig.update_layout(height=1750)
    st.plotly_chart(fig)
