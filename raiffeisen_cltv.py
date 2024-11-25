import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifelines import KaplanMeierFitter

# App Title
st.set_page_config(layout="wide")
st.title("Raiffeisen Bank - Customer Lifetime Value Dashboard")

# File Upload
uploaded_file = st.file_uploader("Upload Customer Transaction Data", type=["csv", "xlsx"])

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

    # Detect Columns
    def detect_column(possible_names):
        for col in data.columns:
            if any(name.lower() in col.lower() for name in possible_names):
                return col
        return None

    # Expected Columns
    expected_columns = {
        "CustomerID": ["customerid", "customer id", "cstid", "id"],
        "Quantity": ["quantity", "qty", "amount"],
        "UnitPrice": ["unitprice", "price per unit", "price"],
        "InvoiceDate": ["invoicedate", "date", "transaction date"]
    }

    detected_columns = {key: detect_column(names) for key, names in expected_columns.items()}
    missing_columns = [key for key, col in detected_columns.items() if col is None]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()

    # Rename columns to standard names
    data = data.rename(columns={v: k for k, v in detected_columns.items()})
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    # Layout Sections
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head(10))

    with col2:
        st.subheader("Data Summary")
        st.write(f"Total Transactions: {len(data)}")
        st.write(f"Total Customers: {data['CustomerID'].nunique()}")
        st.write(f"Total Revenue: ${data['TotalPrice'].sum():,.2f}")

    # EDA Section
    st.markdown("---")
    st.header("Exploratory Data Analysis")

    # Date Range Selector
    min_date, max_date = data["InvoiceDate"].min(), data["InvoiceDate"].max()
    date_range = st.slider(
        "Select Date Range for Analysis",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )
    filtered_data = data[(data["InvoiceDate"] >= date_range[0]) & (data["InvoiceDate"] <= date_range[1])]

    # Sales Over Time
    st.subheader("Sales Over Time")
    sales_over_time = filtered_data.groupby(filtered_data["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    sales_over_time.plot(kind="line", ax=ax)
    ax.set_title("Monthly Sales")
    ax.set_ylabel("Sales Amount")
    st.pyplot(fig)

    # Heatmap
    st.subheader("Sales Heatmap (Day vs Hour)")
    filtered_data["Day"] = filtered_data["InvoiceDate"].dt.day_name()
    filtered_data["Hour"] = filtered_data["InvoiceDate"].dt.hour
    heatmap_data = filtered_data.pivot_table(values="TotalPrice", index="Day", columns="Hour", aggfunc="sum").fillna(0)
    st.dataframe(heatmap_data.style.background_gradient(cmap="viridis"))

    # Prediction Section
    st.markdown("---")
    st.header("Customer Lifetime Value Prediction")

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

    st.subheader("Top Customers by Predicted CLTV")
    st.dataframe(rfm.sort_values("ExpectedRevenue", ascending=False).head(10))

    # Data Manipulation Section
    st.markdown("---")
    st.header("Data Manipulation")

    st.subheader("Filter Data by Customer ID")
    customer_id_filter = st.text_input("Enter Customer ID")
    if customer_id_filter:
        filtered_customer_data = data[data["CustomerID"].astype(str).str.contains(customer_id_filter, na=False)]
        st.write(filtered_customer_data)
    else:
        st.write("No filter applied.")

    # Download Section
    st.markdown("---")
    st.subheader("Download Processed Data")
    st.download_button(
        "Download RFM and CLTV Results",
        rfm.to_csv(index=False).encode("utf-8"),
        "rfm_cltv_results.csv",
        "text/csv",
    )

else:
    st.info("Please upload a file to start.")
