import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("customer_transactions.csv")
data['Date'] = pd.to_datetime(data['Date'])

# Predict CLV based on simple model
def predict_clv(data):
    X = data[['Recency', 'Frequency', 'Monetary']]
    y = data['CLV']
    model = LinearRegression()
    model.fit(X, y)
    data['Predicted_CLV'] = model.predict(X)
    return data

data = predict_clv(data)

# Include TailwindCSS via CDN for styling
st.markdown(
    """
    <style>
        @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    </style>
    """, 
    unsafe_allow_html=True
)

# Add custom CSS for better styling
st.markdown("""
    <style>
        .main-header { 
            font-size: 3em; 
            font-weight: bold; 
            color: #4A90E2; 
        }
        .sub-header { 
            font-size: 2em; 
            font-weight: 600; 
            color: #4A90E2;
        }
        .card {
            background-color: #f3f4f6;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
        }
        .card-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .card-content {
            font-size: 1.1em;
        }
    </style>
""", unsafe_allow_html=True)

# Layout: Main Header
st.markdown('<div class="main-header">Customer Lifetime Value Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Time Series Analysis & Product Insights</div>', unsafe_allow_html=True)

# Sidebar for filters
st.sidebar.markdown('<div class="card"><div class="card-title">Filters</div></div>', unsafe_allow_html=True)
selected_product = st.sidebar.selectbox("Select Product", options=data['Product_ID'].unique())
start_date = st.sidebar.date_input('Start Date', min_value=data['Date'].min(), max_value=data['Date'].max())
end_date = st.sidebar.date_input('End Date', min_value=data['Date'].min(), max_value=data['Date'].max())

# Time-Series Sales over Time
sales_over_time = data.groupby(data['Date'].dt.to_period('M'))['Sales_Amount'].sum().reset_index()
fig = px.line(sales_over_time, x='Date', y='Sales_Amount', title='Sales Over Time')
st.plotly_chart(fig)

# Filter data
filtered_data = data[(data['Product_ID'] == selected_product) & 
                     (data['Date'] >= pd.to_datetime(start_date)) & 
                     (data['Date'] <= pd.to_datetime(end_date))]

# Columns layout for organized sections
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="card"><div class="card-title">Filtered Data</div></div>', unsafe_allow_html=True)
    st.dataframe(filtered_data)

with col2:
    st.markdown('<div class="card"><div class="card-title">Predicted CLV</div></div>', unsafe_allow_html=True)
    st.dataframe(data[['Customer_ID', 'Predicted_CLV']])

# Product Sales Analysis
product_sales = data.groupby('Product_ID')['Sales_Amount'].sum().reset_index()
product_sales_sorted = product_sales.sort_values(by='Sales_Amount', ascending=False)

fig2 = px.bar(product_sales_sorted, x='Product_ID', y='Sales_Amount', title='Total Sales by Product')
st.plotly_chart(fig2)

# Summary: Top and Least Selling Products
st.markdown('<div class="sub-header">Product Sales Overview</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="card"><div class="card-title">Top Products</div></div>', unsafe_allow_html=True)
    st.dataframe(product_sales_sorted.head(10))

with col2:
    st.markdown('<div class="card"><div class="card-title">Least Products</div></div>', unsafe_allow_html=True)
    st.dataframe(product_sales_sorted.tail(10))
