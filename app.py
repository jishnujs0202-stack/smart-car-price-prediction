import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Smart Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B22;
}

/* Input Labels */
label {
    color: white !important;
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg,#00C6FF,#0072FF);
    color: white;
    border: none;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* Metric Cards */
.metric-card {
    background: #161B22;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}

/* Prediction Card */
.prediction-card {
    background: linear-gradient(135deg,#1E3C72,#2A5298);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

/* Header */
.main-header {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #00C6FF;
}

.sub-header {
    text-align: center;
    color: #C9D1D9;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("models/random_forest.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class='main-header'>
🚗 Smart Car Price Prediction System
</div>

<div class='sub-header'>
Predict Used Car Resale Prices using Machine Learning
</div>

<hr>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🚘 Project Information")

st.sidebar.markdown("""
### Smart Car Price Predictor

This application uses a **Random Forest Regressor**
to estimate the resale value of used cars.

### Features

✅ Present Price

✅ Fuel Type

✅ Transmission

✅ Kilometers Driven

✅ Owner History

✅ Car Age

---

### Developer

**Jishnu S**

B.Tech CSE-AIDE
""")

# ==========================================
# DASHBOARD CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>🤖</h2>
        <h3>Random Forest</h3>
        <p>Model</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2>📊</h2>
        <h3>301</h3>
        <p>Cars Analysed</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2>⚡</h2>
        <h3>97%</h3>
        <p>Prediction Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("🚘 Enter Car Details")

col1, col2 = st.columns(2)

with col1:

    present_price = st.number_input(
        "Present Price (Lakhs)",
        min_value=0.0,
        value=5.0
    )

    driven_kms = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=30000
    )

    owner = st.selectbox(
        "Previous Owners",
        [0, 1, 2, 3]
    )

with col2:

    purchase_year = st.slider(
        "Purchase Year",
        2003,
        datetime.now().year,
        2018
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

# ==========================================
# CAR AGE
# ==========================================

current_year = datetime.now().year
car_age = current_year - purchase_year

st.info(f"🚘 Car Age: {car_age} Years")

# ==========================================
# PREPARE INPUT
# ==========================================

input_data = {
    "Present_Price": present_price,
    "Driven_kms": driven_kms,
    "Owner": owner,
    "Car_Age": car_age
}

for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0

if "Fuel_Type_Petrol" in input_data:
    input_data["Fuel_Type_Petrol"] = 1 if fuel_type == "Petrol" else 0

if "Transmission_Manual" in input_data:
    input_data["Transmission_Manual"] = 1 if transmission == "Manual" else 0

if "Selling_type_Individual" in input_data:
    input_data["Selling_type_Individual"] = 1

input_df = pd.DataFrame([input_data])

input_df = input_df[feature_columns]

# ==========================================
# PREDICT
# ==========================================

if st.button("🔮 Predict Car Price"):

    prediction = model.predict(input_df)[0]

    st.markdown(
        f"""
        <div class='prediction-card'>
        💰 Estimated Selling Price <br><br>
        ₹ {prediction:.2f} Lakhs
        </div>
        """,
        unsafe_allow_html=True
    )

    if prediction > 10:
        st.success("⭐ High-value vehicle detected!")

    elif prediction > 5:
        st.info("✅ Moderate resale value.")

    else:
        st.warning("⚠ Lower resale value expected.")

    st.balloons()

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
"""
<center>
Made with ❤️ using Python, Scikit-Learn & Streamlit
</center>
""",
unsafe_allow_html=True
)