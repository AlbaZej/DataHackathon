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

# Create side-by-side boxplots for comparison
def compare_boxplots(data_before, data_after, numeric_cols):
    for col in numeric_cols:
        if col in data_after.columns:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
            data_before[col].plot(kind="box", ax=axes[0], title=f"Before Cleaning: {col}")
            data_after[col].plot(kind="box", ax=axes[1], title=f"After Cleaning: {col}")
            plt.tight_layout()
            st.pyplot(fig)

if uploaded_file:
    # Load the data
    try:
        data = read_file(uploaded_file)
        st.write("## Uploaded Dataset")
        st.dataframe(data)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # EDA Before Cleaning
    st.write("## Exploratory Data Analysis (Before Cleaning)")
    st.write("### Summary Statistics")
    st.write(data.describe())

    # Histograms Before Cleaning
    st.write("### Histograms (Before Cleaning)")
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        st.write(f"Histogram for {col}")
        fig, ax = plt.subplots()
        data[col].hist(ax=ax, bins=20)
        ax.set_title(f"Histogram for {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # Data Cleaning
    st.write("## Data Cleaning")
    st.write("Dropping rows where quantity <= 0, price <= 0, price > 200, and rows with specified stockcodes.")

    # Check required columns exist
    required_columns = ["quantity", "price"]
    if all(col in data.columns for col in required_columns):
        cleaned_data = data[
            (data["quantity"] > 0) &
            (data["price"] > 0) &
            (data["price"] <= 200)
        ]
    else:
        st.write("Dataset does not contain required columns for filtering.")
        cleaned_data = pd.DataFrame()

    # Filter stockcodes if column exists
    if "stockcode" in cleaned_data.columns:
        excluded_stockcodes = ["bankcharges", "c2", "dot", "post"]
        cleaned_data = cleaned_data[~cleaned_data["stockcode"].str.lower().isin(excluded_stockcodes)]
    else:
        st.write("Stockcode column not found in dataset.")

    # Check if cleaned data is valid
    if not cleaned_data.empty:
        st.write("### Cleaned Dataset")
        st.dataframe(cleaned_data)

        # EDA After Cleaning
        st.write("## Exploratory Data Analysis (After Cleaning)")
        st.write("### Summary Statistics")
        st.write(cleaned_data.describe())

        # Compare Boxplots Before and After Cleaning
        st.write("### Boxplots Comparison: Before and After Cleaning")
        compare_boxplots(data, cleaned_data, numeric_cols)

        # Feature Engineering
        st.write("## Feature Engineering")
        if all(col in cleaned_data.columns for col in ["quantity", "price", "customer_id", "invoice_date"]):
            cleaned_data["transaction_value"] = cleaned_data["quantity"] * cleaned_data["price"]
            customer_group = cleaned_data.groupby("customer_id").agg(
                frequency=("invoice_date", "count"),
                avg_transaction_value=("transaction_value", "mean"),
                total_transaction_value=("transaction_value", "sum"),
                length_of_relationship=("invoice_date", lambda x: (x.max() - x.min()).days if len(x) > 1 else 0)
            ).reset_index()
            st.write("### Engineered Features")
            st.dataframe(customer_group)
        else:
            st.warning("Required columns (e.g., 'quantity', 'price', 'customer_id', 'invoice_date') are missing for feature engineering.")

        # Predicting CLTV
        st.write("## CLTV Prediction")
        target_col = st.selectbox("Select the target column (CLTV)", options=customer_group.columns if "customer_group" in locals() else numeric_cols)
        feature_cols = st.multiselect(
            "Select feature columns",
            options=[col for col in customer_group.columns if col != target_col] if "customer_group" in locals() else numeric_cols
        )

        if target_col and feature_cols:
            X = customer_group[feature_cols]
            y = customer_group[target_col]

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
        st.write("Cleaned data is empty after applying filters.")
else:
    st.write("Upload a dataset to get started.")
