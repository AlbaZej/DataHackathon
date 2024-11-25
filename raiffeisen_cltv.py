import streamlit as st
import pandas as pd

# Title
st.title("CLTV Prediction Application")

# File upload
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, Excel, JSON)", 
    type=["csv", "xlsx", "xls", "json"]
)

if uploaded_file is not None:
    try:
        # Load the data
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            data = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format!")
            data = None
        
        if data is not None:
            st.write("Original Data:")
            st.dataframe(data)

            # Section 1: Flag Rows with Price Discrepancies
            st.subheader("1. Flag Products with Large Price Discrepancies")
            price_discrepancy_threshold = st.slider(
                "Set maximum allowed price variation for a single product (%)", 
                min_value=0, 
                max_value=500, 
                value=100
            )

            # Flag rows with price discrepancies
            product_price_stats = data.groupby('StockCode')['Price'].agg(['min', 'max'])
            product_price_stats['price_variation'] = (
                (product_price_stats['max'] - product_price_stats['min']) / product_price_stats['min'] * 100
            )
            flagged_discrepancies = product_price_stats[
                product_price_stats['price_variation'] > price_discrepancy_threshold
            ].index
            flagged_rows_price = data[data['StockCode'].isin(flagged_discrepancies)]

            st.write("Flagged Rows with Price Discrepancies:")
            selected_rows_price = st.experimental_data_editor(flagged_rows_price, num_rows="dynamic")

            # Section 2: Flag Rows with Unwanted StockCode Patterns
            st.subheader("2. Flag Rows with Unwanted StockCode Patterns")
            unwanted_patterns = ["C2", "BANK CHARGES", "AMAZONFEE", "TEST", "GIFT"]
            flagged_rows_stockcode = data[
                data['StockCode'].str.contains('|'.join(unwanted_patterns), case=False, na=False)
            ]

            st.write("Flagged Rows with Unwanted StockCode Patterns:")
            selected_rows_stockcode = st.experimental_data_editor(flagged_rows_stockcode, num_rows="dynamic")

            # Drop Selected Rows
            if st.button("Drop Selected Rows"):
                # Combine flagged rows for dropping
                rows_to_drop = pd.concat([selected_rows_price, selected_rows_stockcode]).index.unique()
                data = data.drop(rows_to_drop)
                st.success(f"Dropped {len(rows_to_drop)} rows.")
                st.write("Cleaned Data:")
                st.dataframe(data)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to start.")
