import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import datetime

# Function to simulate CLV prediction (for the sake of this example)
def predict_clv(data, months):
    # Simple CLV prediction formula (this can be made more complex)
    data['CLV'] = (data['Frequency'] * data['Monetary']) * (1 + 0.1 * months)
    return data

# Function to simulate customer survival rate (using a simple placeholder model)
def predict_survival_rate(data, months):
    # Placeholder: Assume a simple survival rate calculation
    survival_rate = np.random.uniform(0.6, 0.9, len(data)) * (1 - 0.05 * months)  # Survival decreases with time
    data['Survival_Rate'] = survival_rate
    return data

# Create some sample data if no file is uploaded
def create_sample_data():
    np.random.seed(42)  # For reproducibility
    
    # Generate random data for 100 customers
    customer_ids = np.arange(1, 101)
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    
    # Sample data for customers
    data = {
        'Customer_ID': np.random.choice(customer_ids, size=500),
        'Product_ID': np.random.choice(products, size=500),
        'Sales_Amount': np.random.uniform(10, 500, size=500),
        'Frequency': np.random.randint(1, 20, size=500),
        'Monetary': np.random.uniform(50, 1000, size=500),
        'Date': pd.to_datetime(np.random.choice(pd.date_range('2023-01-01', '2023-12-31', freq='D'), size=500)),
        'Anomaly': np.random.choice([True, False], size=500, p=[0.05, 0.95])  # Flagged anomalies
    }
    
    return pd.DataFrame(data)

# Data cleaning functions
def clean_data(data):
    # Remove duplicate rows
    data_cleaned = data.drop_duplicates()
    
    # Fill missing values (e.g., with the mean or median)
    data_cleaned['Sales_Amount'] = data_cleaned['Sales_Amount'].fillna(data_cleaned['Sales_Amount'].median())
    data_cleaned['Frequency'] = data_cleaned['Frequency'].fillna(data_cleaned['Frequency'].mode()[0])
    data_cleaned['Monetary'] = data_cleaned['Monetary'].fillna(data_cleaned['Monetary'].median())
    
    # Ensure data types are correct
    data_cleaned['Customer_ID'] = data_cleaned['Customer_ID'].astype(int)
    data_cleaned['Product_ID'] = data_cleaned['Product_ID'].astype(str)
    
    return data_cleaned

# Generate sample data
data = create_sample_data()

# Add a title
st.title("Customer Analytics Dashboard")

# Sidebar for navigation
sidebar = st.sidebar.radio("Select Page", ("Flagged Data", "Statistics", "Prediction", "Data Cleaning"))

# Filter data by date range (optional feature)
st.sidebar.subheader("Date Range Filter")
min_date, max_date = data['Date'].min(), data['Date'].max()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Filter data by selected date range
filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]
st.sidebar.write(f"Displaying data from {start_date} to {end_date}")

# Display the content based on selected page
if sidebar == "Flagged Data":
    st.subheader("Flagged Data (Anomalies)")
    flagged_data = filtered_data[filtered_data['Anomaly'] == True]
    st.write(f"Found {len(flagged_data)} flagged anomalies")
    
    # Display flagged anomalies
    st.dataframe(flagged_data)
    
    # Insights on anomalies
    st.subheader("Why These Are Anomalies")
    anomaly_insights = []
    
    for idx, row in flagged_data.iterrows():
        if row['Sales_Amount'] > 400:
            anomaly_insights.append(f"Customer {row['Customer_ID']} with Sales Amount {row['Sales_Amount']} is flagged due to unusually high sales.")
        if row['Frequency'] > 15:
            anomaly_insights.append(f"Customer {row['Customer_ID']} with Frequency {row['Frequency']} is flagged due to an unusually high purchase frequency.")
        if row['Monetary'] > 900:
            anomaly_insights.append(f"Customer {row['Customer_ID']} with Monetary {row['Monetary']} is flagged due to unusually high monetary value.")
    
    st.write("\n".join(anomaly_insights))

elif sidebar == "Statistics":
    st.subheader("Statistics & Visualizations")
    
    # Sales Over Time (Line Chart)
    st.subheader("Sales Over Time")
    sales_over_time = data.groupby(data['Date'].dt.to_period('M'))['Sales_Amount'].sum().reset_index()
    sales_over_time['Date'] = sales_over_time['Date'].dt.to_timestamp()  # Convert period to timestamp for plotting
    fig_sales = px.line(sales_over_time, x='Date', y='Sales_Amount', title='Sales Over Time', labels={'Date': 'Date', 'Sales_Amount': 'Sales Amount'})
    st.plotly_chart(fig_sales)
    
    # Total Sales by Product (Bar Chart)
    st.subheader("Total Sales by Product")
    product_sales = filtered_data.groupby('Product_ID')['Sales_Amount'].sum().reset_index()
    fig_product_sales = px.bar(product_sales, x='Product_ID', y='Sales_Amount', title='Total Sales by Product', labels={'Product_ID': 'Product', 'Sales_Amount': 'Total Sales'})
    st.plotly_chart(fig_product_sales)

elif sidebar == "Prediction":
    st.subheader("Customer Lifetime Value (CLV) Prediction")
    
    # Predict CLV for 3, 6, and 9 months
    months_options = [3, 6, 9]
    selected_months = st.selectbox("Select months for CLV Prediction", months_options)
    
    # Predict CLV based on selected months
    clv_data = predict_clv(filtered_data.copy(), selected_months)
    st.write(f"Predicted CLV for {selected_months} months")
    st.dataframe(clv_data[['Customer_ID', 'CLV']].head())

    # Predict Survival Rate for 3, 6, and 9 months
    survival_data = predict_survival_rate(filtered_data.copy(), selected_months)
    st.write(f"Predicted Survival Rate for {selected_months} months")
    st.dataframe(survival_data[['Customer_ID', 'Survival_Rate']].head())

elif sidebar == "Data Cleaning":
    st.subheader("Data Cleaning")
    
    # Show raw data preview before cleaning
    st.write("**Raw Data Preview**")
    st.dataframe(filtered_data.head())
    
    # Clean the data
    st.write("**Cleaning the Data**")
    cleaned_data = clean_data(filtered_data)
    
    # Show cleaned data preview
    st.write("**Cleaned Data Preview**")
    st.dataframe(cleaned_data.head())
    
    # Show message indicating cleaning is done
    st.write("Data has been cleaned. Duplicate rows have been removed, missing values have been filled, and data types have been corrected.")
