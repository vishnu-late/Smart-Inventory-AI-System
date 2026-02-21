import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from utils.preprocessing import clean_data, add_features
from utils.model_utils import classify_product

# Page Config
st.set_page_config(page_title="Smart Inventory AI", layout="wide")

st.title("📊 Smart Inventory Management AI")

# Sidebar for File Upload
st.sidebar.header("Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

@st.cache_data
def load_model():
    try:
        with open('models/model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    
    # Normalize columns
    df.columns = df.columns.str.strip()
    
    # Map common column names
    column_mapping = {
        'date': 'Date', 'Date': 'Date',
        'product_id': 'Product_ID', 'Product ID': 'Product_ID', 'SKU': 'Product_ID', 'Item': 'Product_ID',
        'quantity': 'Quantity', 'Qty': 'Quantity', 'Sales': 'Quantity', 'Demand': 'Quantity',
        'stock': 'Stock', 'Inventory': 'Stock', 'Quantity on Hand': 'Stock'
    }
    
    df.rename(columns=column_mapping, inplace=True)

    # Validation
    required_columns = ['Date', 'Product_ID', 'Quantity', 'Stock']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Uploaded CSV is missing required columns: {missing_columns}")
        st.write(f"Found columns: {list(df.columns)}")
        st.stop()
        
    st.sidebar.success("File Uploaded Successfully!")
else:
    # Load default data if available
    try:
        df = pd.read_csv('data/sales_data.csv')
        st.sidebar.info("Using Sample Data")
    except FileNotFoundError:
        st.error("No data found. Upload a CSV file or generate sample data.")
        st.stop()

# Preprocessing
if df is not None:
    # Basic info
    st.header("📦 Inventory Overview")
    
    # Calculate Metrics
    total_products = df['Product_ID'].nunique()
    total_stock = df['Stock'].sum()
    low_stock_threshold = 20 # Configurable
    low_stock_count = df[df['Stock'] < low_stock_threshold]['Product_ID'].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products", total_products)
    col2.metric("Total Stock", total_stock)
    col3.metric("Low Stock Items", low_stock_count, delta_color="inverse")

    # Demand Forecasting
    st.header("📈 Demand Forecasting (Next 30 Days)")
    
    if model:
        # Prepare data for prediction (using last available date + 30 days)
        last_date = pd.to_datetime(df['Date']).max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
        
        # Create a dataframe for all products for future dates
        products = df['Product_ID'].unique()
        forecast_data = []
        
        for product in products:
            # Simple assumption: Stock remains constant for prediction input (or depletes, but let's keep it constant for simplicity of input features)
            current_stock = df[df['Product_ID'] == product]['Stock'].iloc[-1] 
            
            for date in future_dates:
                forecast_data.append({
                    'Date': date,
                    'Product_ID': product,
                    'Stock': current_stock, # Input feature for model
                    'Day': date.day,
                    'Month': date.month,
                    'Year': date.year,
                    'DayOfWeek': date.dayofweek
                })
                
        forecast_df = pd.DataFrame(forecast_data)
        
        # Predict
        X_pred = forecast_df[['Day', 'Month', 'Year', 'DayOfWeek', 'Stock']]
        forecast_df['Predicted_Demand'] = model.predict(X_pred)
        
        # Aggregate demand per day for visualization
        daily_forecast = forecast_df.groupby('Date')['Predicted_Demand'].sum().reset_index()
        fig = px.line(daily_forecast, x='Date', y='Predicted_Demand', title='Total Predicted Demand Trend')
        st.plotly_chart(fig, use_container_width=True)
        
        # Stock Alert System
        st.header("⚠️ Low Stock Alerts")
        
        # Calculate total predicted demand for next 30 days per product
        product_demand = forecast_df.groupby('Product_ID')['Predicted_Demand'].sum().reset_index()
        
        # Get current stock
        latest_stock = df.drop_duplicates('Product_ID', keep='last')[['Product_ID', 'Stock']]
        
        alert_df = pd.merge(product_demand, latest_stock, on='Product_ID')
        alert_df['Status'] = alert_df.apply(lambda x: "🚨 Low Stock" if x['Predicted_Demand'] > x['Stock'] else "✅ Sufficient", axis=1)
        
        # Classification
        # Calculate avg daily sales from historical data
        hist_sales = df.groupby('Product_ID')['Quantity'].mean().reset_index()
        hist_sales['Movement_Type'] = hist_sales['Quantity'].apply(classify_product)
        
        final_df = pd.merge(alert_df, hist_sales[['Product_ID', 'Movement_Type']], on='Product_ID')
        
        # Display Alerts Table
        st.dataframe(final_df.style.applymap(lambda v: 'color: red;' if v == '🚨 Low Stock' else 'color: green;', subset=['Status']))
        
        # Download Report
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast Report", csv, "forecast_report.csv", "text/csv")
        
    else:
        st.warning("Model not trained yet. Please train the model using `train.py`.")

