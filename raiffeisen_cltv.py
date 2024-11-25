import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set up the app title
st.title("CLTV Prediction App")

# File upload
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV, Excel, or JSON file", type=["csv", "xlsx", "xls", "json"])

# Function to read uploaded file
def read_file(file):
    if file.name.endswith("csv"):
        return pd.read_csv(file)
    elif file.name.endswith(("xlsx", "xls")):
        return pd.read_excel(file)
    elif file.name.endswith("json"):
        return pd.read_json(file)

if uploaded_file:
    # Load the data
    try:
        data = read_file(uploaded_file)
        st.write("## Uploaded Dataset")
        st.dataframe(data)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Data Cleaning
    st.write("## Data Cleaning")
    st.write("Automatically handling missing values and dropping duplicates.")
    data_cleaned = data.drop_duplicates()
    data_cleaned = data_cleaned.fillna(data_cleaned.mean(numeric_only=True))
    st.write("### Cleaned Dataset")
    st.dataframe(data_cleaned)

    # EDA: Summary Statistics
    st.write("## Exploratory Data Analysis (EDA)")
    st.write("### Summary Statistics")
    st.write(data_cleaned.describe())

    # EDA: Histograms
    st.write("### Histograms")
    numeric_cols = data_cleaned.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        st.write(f"Histogram for {col}")
        fig, ax = plt.subplots()
        data_cleaned[col].hist(ax=ax, bins=20)
        ax.set_title(f"Histogram for {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # EDA: Boxplots
    st.write("### Boxplots")
    for col in numeric_cols:
        st.write(f"Boxplot for {col}")
        fig, ax = plt.subplots()
        data_cleaned[col].plot(kind="box", ax=ax)
        ax.set_title(f"Boxplot for {col}")
        st.pyplot(fig)

    # Predicting CLTV
    st.write("## CLTV Prediction")
    target_col = st.selectbox("Select the target column (CLTV)", options=numeric_cols)
    feature_cols = st.multiselect("Select feature columns", options=numeric_cols, default=[col for col in numeric_cols if col != target_col])

    if target_col and feature_cols:
        X = data_cleaned[feature_cols]
        y = data_cleaned[target_col]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Performance")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-Squared: {r2:.2f}")

        # Plot Actual vs Predicted
        st.write("### Actual vs Predicted CLTV")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
        ax.set_title("Actual vs Predicted CLTV")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

else:
    st.write("Upload a dataset to get started.")
