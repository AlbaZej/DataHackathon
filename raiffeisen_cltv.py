import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import datetime

# Function to simulate CLV prediction (for the sake of this example)
def predict_clv(data):
    # Simple CLV calculation based on frequency and monetary values
    data['CLV'] = data['Frequency'] * data['Monetary']
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
        'Date': pd.to_datetime(np.random.choice(pd.date_range('2023-01-01', '2023-12-31', freq='D'), size=500))
    }
    
    return pd.DataFrame(data)

# Generate sample data
data = create_sample_data()

# Show the first few rows of the sample data to the user
st.write("Sample Data Preview:")
st.dataframe(data.head())  # Display the first 5 rows of the data

# Add a title
st.title("Customer Lifetime Value (CLV) Dashboard")

# Time Series Analysis - Sales Over Time
st.subheader("Sales Over Time")
sales_over_time = data.groupby(data['Date'].dt.to_period('M'))['Sales_Amount'].sum().reset_index()
sales_over_time['Date'] = sales_over_time['Date'].dt.to_timestamp()  # Convert period to timestamp for plotting

# Plot sales over time
fig_sales = px.line(sales_over_time, x='Date', y='Sales_Amount', title='Sales Over Time', labels={'Date': 'Date', 'Sales_Amount': 'Sales Amount'})
st.plotly_chart(fig_sales)

# Filter by date range (user input)
st.sidebar.subheader("Filters")
min_date, max_date = data['Date'].min(), data['Date'].max()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Filter data by selected date range
filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]
st.write(f"Displaying data from {start_date} to {end_date}")

# CLV prediction
st.subheader("Customer Lifetime Value (CLV) Prediction")
filtered_data = predict_clv(filtered_data)

# Display predicted CLV
st.dataframe(filtered_data[['Customer_ID', 'CLV']].head())

# Product Sales Analysis
st.subheader("Total Sales by Product")
product_sales = filtered_data.groupby('Product_ID')['Sales_Amount'].sum().reset_index()
fig_product_sales = px.bar(product_sales, x='Product_ID', y='Sales_Amount', title='Total Sales by Product', labels={'Product_ID': 'Product', 'Sales_Amount': 'Total Sales'})
st.plotly_chart(fig_product_sales)

# Top and Least Selling Products
st.subheader("Top and Least Selling Products")

# Sort products by sales
top_products = product_sales.sort_values(by='Sales_Amount', ascending=False).head(10)
least_products = product_sales.sort_values(by='Sales_Amount').head(10)

st.write("Top 10 Best Selling Products:")
st.dataframe(top_products)

st.write("Top 10 Least Selling Products:")
st.dataframe(least_products)
