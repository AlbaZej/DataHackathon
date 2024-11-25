import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifelines import KaplanMeierFitter

# App Title
st.title("Raiffeisen Bank - Customer Lifetime Value (CLTV) Prediction")

# File Upload
uploaded_file = st.file_uploader("Upload Customer Transaction Data (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Check file type and load data accordingly
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    
    # Display uploaded data
    st.subheader("Uploaded Data")
    st.write(data.head())
    
    # Data Preprocessing
    st.subheader("Data Preprocessing")
    data = data.dropna(subset=['CustomerID'])
    data = data[data['Quantity'] > 0]
    data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    st.write("### Cleaned Data Preview")
    st.write(data.head())

    # RFM Metrics
    st.subheader("RFM Analysis")
    snapshot_date = data['InvoiceDate'].max()
    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})
    st.write("### RFM Table")
    st.write(rfm.head())

    # Kaplan-Meier Survival Analysis
    st.subheader("Survival Analysis")
    rfm['Churned'] = (rfm['Recency'] > 180).astype(int)
    rfm['TimeToChurn'] = rfm['Recency']
    kmf = KaplanMeierFitter()
    kmf.fit(rfm['TimeToChurn'], event_observed=rfm['Churned'])
    plt.figure(figsize=(10, 5))
    kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve")
    st.pyplot(plt)

    # CLTV Prediction
    st.subheader("CLTV Prediction")
    bgf = BetaGeoFitter()
    bgf.fit(rfm['Frequency'], rfm['Recency'], rfm['Monetary'])
    ggf = GammaGammaFitter()
    ggf.fit(rfm['Frequency'], rfm['Monetary'])
    rfm['ExpectedRevenue'] = ggf.customer_lifetime_value(
        bgf, rfm['Frequency'], rfm['Recency'], rfm['Monetary'], time=12, freq='D'
    )
    rfm['SurvivalRate'] = kmf.predict(rfm['TimeToChurn'])
    rfm['AdjustedCLTV'] = rfm['ExpectedRevenue'] * rfm['SurvivalRate']
    
    st.write("### Top Customers by Adjusted CLTV")
    st.write(rfm.sort_values('AdjustedCLTV', ascending=False).head(10))

    # Download Results
    def convert_df(df):
        # Converts dataframe to CSV or Excel for download
        return df.to_csv(index=True).encode('utf-8')

    st.download_button(
        "Download RFM and CLTV Results (CSV)",
        convert_df(rfm),
        "rfm_cltv_results.csv",
        "text/csv",
        key='download-csv'
    )
else:
    st.info("Please upload a CSV or Excel file to start.")
