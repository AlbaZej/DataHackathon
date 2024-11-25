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
    if file_extension in ["csv"]:
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
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    st.write("### Cleaned Data Preview")
    st.write(data.head())

    # RFM Metrics
    st.subheader("RFM Analysis")
    snapshot_date = data["InvoiceDate"].max()
    rfm = data.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "CustomerID": "count",
        "TotalPrice": "sum"
    }).rename(columns={"InvoiceDate": "Recency", "CustomerID": "Frequency", "TotalPrice": "Monetary"})
    rfm["T"] = rfm["Recency"] + 30  # Assumed snapshot duration
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
    # CLTV Prediction
st.subheader("CLTV Prediction")
valid_rfm = rfm[(rfm["Frequency"] > 0) & (rfm["Recency"] > 0) & (rfm["T"] > 0) & (rfm["Monetary"] > 0)]

if valid_rfm.empty:
    st.error("No valid data for CLTV prediction. All monetary values are non-positive. Please check the input data.")
else:
    try:
        # Fit the Beta-Geometric/NBD model
        bgf = BetaGeoFitter(penalizer_coef=0.01)  # Adding a small penalizer for stability
        bgf.fit(valid_rfm["Frequency"], valid_rfm["Recency"], valid_rfm["T"])

        # Fit the Gamma-Gamma model
        ggf = GammaGammaFitter(penalizer_coef=0.01)
        ggf.fit(valid_rfm["Frequency"], valid_rfm["Monetary"])

        # Predict CLTV
        valid_rfm["ExpectedRevenue"] = ggf.customer_lifetime_value(
            bgf,
            valid_rfm["Frequency"],
            valid_rfm["Recency"],
            valid_rfm["T"],
            valid_rfm["Monetary"],  # Pass the monetary values
            time=12,  # Predict for the next 12 months
            freq="D"  # Data frequency is daily
        )

        # Combine Survival Analysis
        valid_rfm["SurvivalRate"] = kmf.predict(valid_rfm["TimeToChurn"])
        valid_rfm["AdjustedCLTV"] = valid_rfm["ExpectedRevenue"] * valid_rfm["SurvivalRate"]

        st.write("### Top Customers by Adjusted CLTV")
        st.write(valid_rfm.sort_values("AdjustedCLTV", ascending=False).head(10))
    except Exception as e:
        st.error(f"CLTV Prediction failed: {e}")
