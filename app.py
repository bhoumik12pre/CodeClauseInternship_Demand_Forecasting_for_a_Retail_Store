import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import datetime

# Step 1: Load Datasets
features = pd.read_csv('D:/Virtual_Internship/Task1/features.csv')
stores = pd.read_csv('D:/Virtual_Internship/Task1/stores.csv')
train = pd.read_csv('D:/Virtual_Internship/Task1/train.csv')
test = pd.read_csv('D:/Virtual_Internship/Task1/test.csv')

# Clean column names by stripping spaces
for df in [features, stores, train, test]:
    df.columns = df.columns.str.strip()

# Debug: Print unique store IDs before merging
st.write("Unique Store IDs in TRAIN:", np.sort(train['Store'].unique()))
st.write("Unique Store IDs in STORES:", np.sort(stores['Store'].unique()))
st.write("Unique Store IDs in FEATURES:", np.sort(features['Store'].unique()))

# Step 2: Merge Datasets using left joins so that all train rows are kept.
df = train.merge(stores, on='Store', how='left').merge(features, on=['Store', 'Date'], how='left')

# Convert Date column and drop rows with invalid dates.
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df = df.sort_values(['Store', 'Date'])
df.drop_duplicates(subset=['Date', 'Store'], inplace=True)

# Convert only necessary columns to numeric.
df['Weekly_Sales'] = pd.to_numeric(df['Weekly_Sales'], errors='coerce')
df['Store'] = pd.to_numeric(df['Store'], errors='coerce')

# Debug: Print unique store IDs after merging
available_store_ids = np.sort(df['Store'].unique())
st.write("Unique Store IDs after merging:", available_store_ids)

# Step 3: Set Date as index and assign weekly frequency per store
df.set_index('Date', inplace=True)

# Group by Store and reindex each group separately
dfs = []
for store, group in df.groupby('Store'):
    group = group.sort_index()
    group = group.asfreq('W-FRI')  # Set weekly frequency for each store independently
    dfs.append(group)
df = pd.concat(dfs)

# Step 4: Feature Engineering
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Week'] = df.index.isocalendar().week
df['Day'] = df.index.day

# Streamlit UI
st.title("Retail Store Sales Forecasting")
st.write("Available Store IDs after merging:", available_store_ids)

min_date = df.index.min().date()
max_date = df.index.max().date()
st.write(f"Data Date Range: {min_date} to {max_date}")

# Use the smallest available store as default if available
if available_store_ids.size > 0:
    default_store = int(available_store_ids[0])
    max_store_id = int(available_store_ids[-1])
else:
    default_store = 1
    max_store_id = 1

store_id = st.number_input("Enter Store ID", min_value=default_store, max_value=max_store_id, value=default_store)

# Date input for train/test split cutoff
cutoff_date = st.date_input("Select cutoff date for train/test split", value=min_date, min_value=min_date, max_value=max_date)

if st.button("Forecast Sales"):
    # Step 5: Data Splitting for the selected store
    store_data = df[df['Store'] == store_id][['Weekly_Sales']]
    if store_data.empty:
        st.error(f"No data available for Store ID {store_id}.")
    else:
        cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")
        train_data = store_data[:cutoff_date_str]
        test_data = store_data[cutoff_date_str:]
        
        if len(test_data) == 0:
            st.error("Not enough test data available for forecasting. Please choose an earlier cutoff date.")
        else:
            # Step 6: Model Training (SARIMA)
            model = SARIMAX(train_data['Weekly_Sales'],
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 52),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            
            # Step 7: Forecasting
            forecast = model_fit.forecast(steps=len(test_data))
            
            # Step 8: Evaluation
            mae = mean_absolute_error(test_data['Weekly_Sales'], forecast)
            rmse = np.sqrt(mean_squared_error(test_data['Weekly_Sales'], forecast))
            st.write(f'MAE: {mae}, RMSE: {rmse}')
            
            # Step 9: Visualization
            st.line_chart({
                "Training Data": train_data['Weekly_Sales'],
                "Actual Sales": test_data['Weekly_Sales'],
                "Predicted Sales": forecast
            })
