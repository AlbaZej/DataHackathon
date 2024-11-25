import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifelines import KaplanMeierFitter

# App Title
st.title("Raiffeisen Bank - Customer Lifetime Value Prediction")

# File Upload
uploaded_file = st.file_uploader("Upload Customer Transaction Data (CSV, Excel, or other formats)", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Load data
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "csv":
        data = pd.read_csv(uploaded_file)
    elif file_extension in ["xlsx", "xls"]:
        import openpyxl
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()
    
    st.subheader("Uploaded Data")
    st.write(data.head())

    # Column Detection
    def detect_column(possible_names):
        """Detect the column based on possible name patterns."""
        for col in data.columns:
            if any(name.lower() in col.lower() for name in possible_names):
                return col
        return None

    # Expected columns and their patterns
    expected_columns = {
        "CustomerID": ["customerid", "customer id", "cstid", "id"],
        "Quantity": ["quantity", "qty", "amount"],
        "UnitPrice": ["unitprice", "price per unit", "price"],
        "InvoiceDate": ["invoicedate", "date", "transaction date"]
    }

    detected_columns = {key: detect_column(names) for key, names in expected_columns.items()}

    # Check for missing columns
    missing_columns = [key for key, col in detected_columns.items() if col is None]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()

    # Rename columns to standard names
    data = data.rename(columns={v: k for k, v in detected_columns.items()})

    # Data Preprocessing
    st.subheader("Data Preprocessing")
    data = data.dropna(subset=["CustomerID"])
    data = data[data["Quantity"] > 0]
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    st.write("### Cleaned Data Preview")
    st.write(data.head())

    # RFM Metrics
    st.subheader("RFM Analysis")
    snapshot_date = data["InvoiceDate"].max()
    rfm = data.groupby("CustomerID").agg({
        "InvoiceDate": [
            lambda x: (snapshot_date - x.max()).days,  # Recency: Days since last purchase
            lambda x: (snapshot_date - x.min()).days  # T: Days since first purchase
        ],
        "CustomerID": "count",  # Frequency: Number of transactions
        "TotalPrice": "sum"     # Monetary: Total spend
    }).reset_index()

    # Rename columns
    rfm.columns = ["CustomerID", "Recency", "T", "Frequency", "Monetary"]

    # Filter for valid rows
    rfm = rfm[rfm["Recency"] <= rfm["T"]]  # Ensure Recency is not greater than T
    rfm = rfm[rfm["Frequency"] > 0]        # Remove customers with no transactions
    rfm = rfm[rfm["Monetary"] > 0]         # Remove customers with no spending

    st.write(rfm.head())

    # Kaplan-Meier Survival Analysis
    st.subheader("Survival Analysis")
    rfm["Churned"] = (rfm["Recency"] > 180).astype(int)
    rfm["TimeToChurn"] = rfm["Recency"]
    kmf = KaplanMeierFitter()
    kmf.fit(rfm["TimeToChurn"], event_observed=rfm["Churned"])
    plt.figure(figsize=(10, 5))
    kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve")
    st.pyplot(plt)

    # CLTV Prediction
    st.subheader("CLTV Prediction")
    bgf = BetaGeoFitter()
    bgf.fit(rfm["Frequency"], rfm["Recency"], rfm["T"])
    ggf = GammaGammaFitter()
    ggf.fit(rfm["Frequency"], rfm["Monetary"])
    rfm["ExpectedRevenue"] = ggf.customer_lifetime_value(
        bgf, rfm["Frequency"], rfm["Recency"], rfm["T"], time=12, freq="D"
    )
    rfm["SurvivalRate"] = kmf.predict(rfm["TimeToChurn"])
    rfm["AdjustedCLTV"] = rfm["ExpectedRevenue"] * rfm["SurvivalRate"]

    st.write("### Top Customers by Adjusted CLTV")
    st.write(rfm.sort_values("AdjustedCLTV", ascending=False).head(10))

    # Download Results
    st.download_button(
        "Download RFM and CLTV Results",
        rfm.to_csv(index=True).encode("utf-8"),
        "rfm_cltv_results.csv",
        "text/csv",
        key="download-csv"
    )
else:
    st.info("Please upload a CSV or Excel file to start.")
