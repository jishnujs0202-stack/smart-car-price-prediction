import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("models/random_forest.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Smart Car Price Prediction System")

st.markdown("Predict the resale value of a used car using Machine Learning")

# Sidebar
st.sidebar.header("Enter Car Details")

present_price = st.sidebar.number_input(
    "Present Price (Lakhs)",
    min_value=0.0,
    value=5.0
)

kms_driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000
)

owner = st.sidebar.selectbox(
    "Number of Owners",
    [0,1,2,3]
)

car_age = st.sidebar.slider(
    "Car Age",
    0,
    20,
    5
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    ["Petrol","Diesel"]
)

seller_type = st.sidebar.selectbox(
    "Seller Type",
    ["Dealer","Individual"]
)

transmission = st.sidebar.selectbox(
    "Transmission",
    ["Manual","Automatic"]
)

input_data = {
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Owner": owner,
    "Car_Age": car_age
}

for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0

if "Fuel_Type_Petrol" in input_data:
    input_data["Fuel_Type_Petrol"] = 1 if fuel_type=="Petrol" else 0

if "Seller_Type_Individual" in input_data:
    input_data["Seller_Type_Individual"] = 1 if seller_type=="Individual" else 0

if "Transmission_Manual" in input_data:
    input_data["Transmission_Manual"] = 1 if transmission=="Manual" else 0

input_df = pd.DataFrame([input_data])

input_df = input_df[feature_columns]

if st.button("Predict Price"):

    prediction = model.predict(input_df)

    st.success(
        f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs"
    )

    st.balloons()

# Dashboard Section
st.subheader("📊 Project Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algorithm", "Random Forest")

with col2:
    st.metric("Dataset", "Car Dataset")

with col3:
    st.metric("Output", "Price Prediction")