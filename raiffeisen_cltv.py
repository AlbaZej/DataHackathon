import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifelines import KaplanMeierFitter

# App Title
st.title("Raiffeisen Bank - Customer Lifetime Value Dashboard")

# Sidebar for Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Upload Data", "EDA", "Prediction", "Data Manipulation"])

# Global Variables
uploaded_file = st.sidebar.file_uploader("Upload Customer Transaction Data", type=["csv", "xlsx"])

# Helper functions
def detect_column(data, possible_names):
    """Detect the column based on possible name patterns."""
    for col in data.columns:
        if any(name.lower() in col.lower() for name in possible_names):
            return col
    return None

# Upload Data Section
if options == "Upload Data":
    st.header("Upload Customer Data")
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file)
        elif file_extension in ["xlsx", "xls"]:
            import openpyxl
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        st.subheader("Uploaded Data Preview")
        st.write(data.head())

# Preprocess and Detect Columns
if uploaded_file:
    if "data" not in locals():
        st.error("Upload a valid file first.")
        st.stop()

    # Expected columns and their patterns
    expected_columns = {
        "CustomerID": ["customerid", "customer id", "cstid", "id"],
        "Quantity": ["quantity", "qty", "amount"],
        "UnitPrice": ["unitprice", "price per unit", "price"],
        "InvoiceDate": ["invoicedate", "date", "transaction date"]
    }

    detected_columns = {key: detect_column(data, names) for key, names in expected_columns.items()}
    missing_columns = [key for key, col in detected_columns.items() if col is None]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()

    # Rename columns to standard names
    data = data.rename(columns={v: k for k, v in detected_columns.items()})
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

# EDA Section
if options == "EDA":
    if uploaded_file:
        st.header("Exploratory Data Analysis")

        # Date Range Selector
        min_date, max_date = data["InvoiceDate"].min(), data["InvoiceDate"].max()
        start_date, end_date = st.date_input(
            "Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
        )
        filtered_data = data[(data["InvoiceDate"] >= pd.Timestamp(start_date)) & (data["InvoiceDate"] <= pd.Timestamp(end_date))]

        # Plot Sales Over Time
        sales_over_time = filtered_data.groupby(filtered_data["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum()
        st.subheader("Sales Over Time")
        plt.figure(figsize=(10, 5))
        sales_over_time.plot(kind="line", title="Monthly Sales")
        plt.ylabel("Sales Amount")
        st.pyplot(plt)

        # Heatmap of Sales by Day and Hour
        filtered_data["Day"] = filtered_data["InvoiceDate"].dt.day_name()
        filtered_data["Hour"] = filtered_data["InvoiceDate"].dt.hour
        sales_heatmap = filtered_data.pivot_table(values="TotalPrice", index="Day", columns="Hour", aggfunc="sum").fillna(0)
        st.subheader("Sales Heatmap (Day vs Hour)")
        st.write(sales_heatmap)

# Prediction Section
if options == "Prediction":
    if uploaded_file:
        st.header("Customer Lifetime Value Prediction")

        # Preprocessing for Prediction
        rfm = data.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (data["InvoiceDate"].max() - x.max()).days),
            Frequency=("CustomerID", "count"),
            Monetary=("TotalPrice", "sum")
        ).reset_index()
        rfm["T"] = rfm["Recency"] + 30

        # Fit Models and Predict
        bgf = BetaGeoFitter(penalizer_coef=0.01)
        bgf.fit(rfm["Frequency"], rfm["Recency"], rfm["T"])
        ggf = GammaGammaFitter(penalizer_coef=0.01)
        ggf.fit(rfm["Frequency"], rfm["Monetary"])
        rfm["ExpectedRevenue"] = ggf.customer_lifetime_value(
            bgf, rfm["Frequency"], rfm["Recency"], rfm["T"], rfm["Monetary"], time=12, freq="D"
        )

        st.subheader("Predicted CLTV")
        st.write(rfm.sort_values("ExpectedRevenue", ascending=False).head(10))

# Data Manipulation Section
if options == "Data Manipulation":
    if uploaded_file:
        st.header("Data Manipulation")

        # Filter Data
        st.subheader("Filter Data")
        customer_id_filter = st.text_input("Filter by Customer ID")
        if customer_id_filter:
            filtered_data = data[data["CustomerID"].astype(str).str.contains(customer_id_filter, na=False)]
            st.write(filtered_data)
        else:
            st.write(data)
