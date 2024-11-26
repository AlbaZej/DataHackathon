import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifetimes import GammaGammaFitter
from lifetimes import BetaGeoFitter

st.title("CLTV Prediction App (Probabilistic Model)")

@st.cache_data
def read_file(file):
    if file.name.endswith("csv"):
        return pd.read_csv(file)
    elif file.name.endswith(("xlsx", "xls")):
        return pd.read_excel(file)
    elif file.name.endswith("json"):
        return pd.read_json(file)

def compare_boxplots(data_before, data_after, numeric_cols):
    for col in numeric_cols:
        if col in data_after.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            data_before[col].plot(kind="box", ax=axes[0], title=f"Before Cleaning: {col}")
            data_after[col].plot(kind="box", ax=axes[1], title=f"After Cleaning: {col}")
            axes[0].set_ylabel("Values")
            plt.tight_layout()
            st.pyplot(fig)

def plot_histograms(data, title):
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        st.write(f"Histogram for {col} ({title})")
        fig, ax = plt.subplots()
        data[col].hist(ax=ax, bins=20)
        ax.set_title(f"{col} ({title})")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

uploaded_file = st.sidebar.file_uploader("Upload a CSV, Excel, or JSON file", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    try:
        data = read_file(uploaded_file)
        st.write("## Uploaded Dataset")
        st.dataframe(data)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.write("## EDA Before Cleaning")
    st.write("### Summary Statistics")
    st.write(data.describe())
    plot_histograms(data, "Before Cleaning")

    st.write("## Data Cleaning")
    required_columns = ["quantity", "price"]
    if all(col in data.columns for col in required_columns):
        cleaned_data = data[
            (data["quantity"] > 0) & 
            (data["price"] > 0) & 
            (data["price"] <= 200)
        ]
    else:
        st.write("Dataset does not contain required columns.")
        cleaned_data = pd.DataFrame()

    if "stockcode" in cleaned_data.columns:
        excluded_stockcodes = ["bankcharges", "c2", "dot", "post"]
        cleaned_data = cleaned_data[~cleaned_data["stockcode"].str.lower().isin(excluded_stockcodes)]
    else:
        st.write("Stockcode column not found in dataset.")

    if not cleaned_data.empty:
        st.write("### Cleaned Dataset")
        st.dataframe(cleaned_data)

        st.write("## EDA After Cleaning")
        st.write("### Summary Statistics")
        st.write(cleaned_data.describe())
        plot_histograms(cleaned_data, "After Cleaning")

        st.write("### Boxplots Comparison")
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        compare_boxplots(data, cleaned_data, numeric_cols)

        st.write("## Feature Engineering")
        if all(col in cleaned_data.columns for col in ["quantity", "price", "customer_id", "invoice_date"]):
            cleaned_data["transaction_value"] = cleaned_data["quantity"] * cleaned_data["price"]

            customer_group = cleaned_data.groupby("customer_id").agg(
                frequency=("invoice_date", "count"),
                avg_transaction_value=("transaction_value", "mean"),
                total_transaction_value=("transaction_value", "sum")
            ).reset_index()
            
            # Filter customers with at least 2 transactions for probabilistic modeling
            customer_group = customer_group[customer_group["frequency"] > 1]

            st.write("### Engineered Features")
            st.dataframe(customer_group)

            st.write("## CLTV Prediction Using Gamma-Gamma Model")
            
            # Fit the Gamma-Gamma model
            ggf = GammaGammaFitter(penalizer_coef=0.01)
            ggf.fit(customer_group["frequency"], customer_group["avg_transaction_value"])

            # Estimate monetary value (CLTV)
            customer_group["predicted_cltv"] = ggf.conditional_expected_average_profit(
                customer_group["frequency"], 
                customer_group["avg_transaction_value"]
            )

            st.write("### Predicted CLTV")
            st.dataframe(customer_group[["customer_id", "predicted_cltv"]])

            st.write("### Gamma-Gamma Model Parameters")
            st.write(f"p: {ggf.params_['p']:.2f}, q: {ggf.params_['q']:.2f}, v: {ggf.params_['v']:.2f}")

            # Visualize the distribution of predicted CLTV
            fig, ax = plt.subplots()
            customer_group["predicted_cltv"].hist(ax=ax, bins=20)
            ax.set_title("Predicted CLTV Distribution")
            ax.set_xlabel("Predicted CLTV")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        else:
            st.warning("Required columns (e.g., 'quantity', 'price', 'customer_id', 'invoice_date') are missing for feature engineering.")

    else:
        st.write("Cleaned dataset is empty. Check cleaning criteria.")
else:
    st.write("Upload a dataset to get started.")
