import streamlit as st
import pandas as pd
import re

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

            # Price Discrepancy Flagging
            st.subheader("Flag Products with Big Price Discrepancies")
            price_discrepancy_threshold = st.slider(
                "Set maximum allowed price variation for a single product (%)", 
                min_value=0, 
                max_value=500, 
                value=100
            )

            product_price_stats = data.groupby('StockCode')['Price'].agg(['min', 'max'])
            product_price_stats['price_variation'] = (
                (product_price_stats['max'] - product_price_stats['min']) / product_price_stats['min'] * 100
            )
            flagged_discrepancies = product_price_stats[
                product_price_stats['price_variation'] > price_discrepancy_threshold
            ].index

            flagged_rows = data[data['StockCode'].isin(flagged_discrepancies)]
            st.write(f"Flagged rows with price discrepancies over {price_discrepancy_threshold}%:")
            st.dataframe(flagged_rows)

            rows_to_drop_price = st.multiselect(
                "Select rows to drop based on price discrepancies:",
                options=flagged_rows.index,
                format_func=lambda x: f"Row {x}"
            )

            # Drop selected rows for price discrepancies
            if st.button("Drop Selected Rows for Price Discrepancies"):
                data = data.drop(rows_to_drop_price)
                st.success(f"Dropped {len(rows_to_drop_price)} rows based on price discrepancies.")

            # StockCode Pattern Flagging
            st.subheader("Flag StockCodes with Unwanted Patterns")
            unwanted_patterns = ["C2", "BANK CHARGES", "AMAZONFEE", "TEST", "GIFT"]
            flagged_stockcode_rows = data[
                data['StockCode'].str.contains('|'.join(unwanted_patterns), case=False, na=False)
            ]
            st.write("Flagged rows with unwanted StockCode patterns:")
            st.dataframe(flagged_stockcode_rows)

            rows_to_drop_stockcode = st.multiselect(
                "Select rows to drop based on StockCode patterns:",
                options=flagged_stockcode_rows.index,
                format_func=lambda x: f"Row {x}"
            )

            # Drop selected rows for StockCode patterns
            if st.button("Drop Selected Rows for StockCode Patterns"):
                data = data.drop(rows_to_drop_stockcode)
                st.success(f"Dropped {len(rows_to_drop_stockcode)} rows based on StockCode patterns.")

            # Display final cleaned data
            st.subheader("Cleaned Data")
            st.dataframe(data)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to start.")
