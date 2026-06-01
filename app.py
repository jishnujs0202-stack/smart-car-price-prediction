import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Smart Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# --------------------------
# CSS
# --------------------------

st.markdown("""
<style>
.stApp{
background:#0E1117;
color:white;
}

section[data-testid="stSidebar"]{
background:#161B22;
}

.metric-card{
background:#161B22;
padding:20px;
border-radius:15px;
text-align:center;
}

.prediction-card{
background:linear-gradient(135deg,#1E3C72,#2A5298);
padding:25px;
border-radius:15px;
text-align:center;
font-size:30px;
font-weight:bold;
color:white;
}

.main-header{
text-align:center;
font-size:45px;
font-weight:bold;
color:#00C6FF;
}

.sub-header{
text-align:center;
color:#C9D1D9;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# LOAD MODEL & DATA
# --------------------------

def load_model():
    model = joblib.load("models/random_forest.pkl")
    cols = joblib.load("feature_columns.pkl")
    return model, cols


@st.cache_data
def load_data():
    df = pd.read_csv("dataset/car_data.csv")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


model, feature_columns = load_model()
df_raw = load_data()


# Auto-detect selling price column
price_col = None

for col in df_raw.columns:
    if "selling" in str(col).lower() and "price" in str(col).lower():
        price_col = col
        break

if price_col is None:
    st.error(f"Selling price column not found. Columns found: {df_raw.columns.tolist()}")
    st.stop()

# --------------------------
# HEADER
# --------------------------

st.markdown("""
<div class='main-header'>
🚗 Smart Car Price Prediction
</div>

<div class='sub-header'>
Machine Learning Based Used Car Valuation
</div>
<hr>
""", unsafe_allow_html=True)

# --------------------------
# SIDEBAR
# --------------------------

st.sidebar.title("Project Information")

st.sidebar.info("""
Model: Random Forest Regressor

Performance:
- R² Score: 96.97%
- MAE: ₹69,612.87
- RMSE: ₹140,949.71

Features:
- KM Driven
- Fuel Type
- Seller Type
- Transmission
- Owner
- Mileage
- Engine
- Max Power
- Seats
- Car Age
""")

# --------------------------
# DASHBOARD
# --------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class='metric-card'>
    <h3>🤖 Random Forest</h3>
    <p>Model</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='metric-card'>
    <h3>8128 TEST</h3>
    <p>Cars</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='metric-card'>
    <h3>96.97%</h3>
    <p>R² Score</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class='metric-card'>
    <h3>₹69.6K</h3>
    <p>MAE</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# TABS
# --------------------------

tab1, tab2, tab3 = st.tabs([
    "Predict Price",
    "EDA",
    "Dataset"
])

# --------------------------
# PREDICTION TAB
# --------------------------

with tab1:

    st.subheader("Enter Car Details")

    col1, col2 = st.columns(2)

    with col1:
        km_driven = st.number_input(
            "KM Driven",
            min_value=0,
            value=50000
        )

        mileage = st.number_input(
            "Mileage",
            min_value=0.0,
            value=20.0
        )

        engine = st.number_input(
            "Engine CC",
            min_value=500,
            value=1200
        )

        seats = st.number_input(
            "Seats",
            min_value=2,
            max_value=10,
            value=5
        )

    with col2:
        purchase_year = st.slider(
            "Year",
            1994,
            datetime.now().year,
            2018
        )

        fuel = st.selectbox(
            "Fuel",
            ["Petrol", "Diesel", "CNG", "LPG"]
        )

        seller_type = st.selectbox(
            "Seller Type",
            ["Dealer", "Individual", "Trustmark Dealer"]
        )

        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

        owner = st.selectbox(
            "Owner",
            [
                "First Owner",
                "Second Owner",
                "Third Owner",
                "Fourth & Above Owner"
            ]
        )

        max_power = st.number_input(
            "Max Power",
            min_value=20.0,
            value=75.0
        )

    car_age = datetime.now().year - purchase_year

    st.info(f"Car Age: {car_age} years")
if st.button("Predict Price"):

    input_data = {col: 0 for col in feature_columns}

    # Numerical Features
    input_data["km_driven"] = km_driven
    input_data["mileage"] = mileage
    input_data["engine"] = engine
    input_data["max_power"] = max_power
    input_data["seats"] = seats
    input_data["Car_Age"] = car_age

    # Fuel
    fuel_feature = f"fuel_{fuel}"
    if fuel_feature in input_data:
        input_data[fuel_feature] = 1

    # Seller Type
    seller_feature = f"seller_type_{seller_type}"
    if seller_feature in input_data:
        input_data[seller_feature] = 1

    # Transmission
    transmission_feature = f"transmission_{transmission}"
    if transmission_feature in input_data:
        input_data[transmission_feature] = 1

    # Owner
    owner_feature = f"owner_{owner}"
    if owner_feature in input_data:
        input_data[owner_feature] = 1

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    prediction = max(0, prediction)

    lower = prediction * 0.90
    upper = prediction * 1.10

    st.markdown(
        f"""
        <div class='prediction-card'>
        Estimated Price<br><br>
        ₹ {prediction:,.0f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        f"Estimated Range: ₹ {lower:,.0f} - ₹ {upper:,.0f}"
    )

# --------------------------
# EDA
# --------------------------

with tab2:

    st.subheader("Dataset Analysis")

    fig, ax = plt.subplots()
    ax.hist(df_raw[price_col], bins=30)
    ax.set_title("Selling Price Distribution")
    ax.set_xlabel("Selling Price")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    if "fuel" in df_raw.columns:
        fuel_avg = (
            df_raw.groupby("fuel")[price_col]
            .mean()
            .sort_values()
        )

        st.subheader("Average Price by Fuel Type")
        st.bar_chart(fuel_avg)

    if "transmission" in df_raw.columns:
        trans_avg = (
            df_raw.groupby("transmission")[price_col]
            .mean()
            .sort_values()
        )

        st.subheader("Average Price by Transmission")
        st.bar_chart(trans_avg)

# --------------------------
# DATASET
# --------------------------

with tab3:

    st.subheader("Dataset Preview")

    st.dataframe(
        df_raw,
        width="stretch"
    )

    st.write("Shape:", df_raw.shape)

    st.subheader("Missing Values")

    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    st.dataframe(
        missing,
        width="stretch",
        hide_index=True
    )

# --------------------------
# FOOTER
# --------------------------

st.markdown("---")

st.markdown(
"""
<center>
Made with ❤️ using Streamlit & Scikit-Learn
</center>
""",
unsafe_allow_html=True
)
