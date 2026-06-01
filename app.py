import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

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

/* Range Card */
.range-card {
    background: #1C2333;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: #C9D1D9;
    margin-top: 10px;
    font-size: 16px;
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

/* Info box */
.info-box {
    background: #161B22;
    padding: 15px 20px;
    border-left: 4px solid #00C6FF;
    border-radius: 8px;
    margin: 10px 0;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL & DATA (with error handling)
# ==========================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/random_forest.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, feature_columns
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Please make sure 'models/random_forest.pkl' and 'feature_columns.pkl' exist.")
        st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("car_data.csv")
        return df
    except FileNotFoundError:
        return None

model, feature_columns = load_model()
df_raw = load_data()

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

### Features Used

✅ Present Price

✅ Fuel Type

✅ Transmission

✅ Kilometers Driven

✅ Owner History

✅ Car Age

---

### Developer

**J S Jishnu**

B.Tech CSE-AIDE

---
### Dataset
- **Source:** Car Dekho
- **Records:** 301 cars
- **Target:** Selling Price (Lakhs ₹)
""")

# ==========================================
# DASHBOARD CARDS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>🤖</h2>
        <h3>Random Forest</h3>
        <p>ML Model</p>
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
        <p>Test Accuracy (R²)</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h2>🎯</h2>
        <h3>6</h3>
        <p>Input Features</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Price",
    "📊 Data Analysis (EDA)",
    "🤖 Model Performance",
    "📋 Dataset Preview"
])

# ==========================================
# TAB 1 — PREDICT
# ==========================================

with tab1:

    st.subheader("🚘 Enter Car Details")

    col1, col2 = st.columns(2)

    with col1:
        present_price = st.number_input(
            "Present Price (Lakhs)",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

        driven_kms = st.number_input(
            "Kilometers Driven",
            min_value=0,
            value=30000,
            step=1000
        )

        owner = st.selectbox(
            "Previous Owners",
            [0, 1, 2, 3],
            format_func=lambda x: f"{x} owner(s)"
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
            ["Petrol", "Diesel", "CNG"]
        )

        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

    # Car Age
    current_year = datetime.now().year
    car_age = current_year - purchase_year

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(f"🚘 Car Age: **{car_age} Years**")
    with col_b:
        st.info(f"📅 Purchase Year: **{purchase_year}**")
    with col_c:
        depreciation = round((car_age * 0.08) * 100, 1)
        st.info(f"📉 Est. Depreciation: **~{min(depreciation, 80)}%**")

    # ---- Prepare Input ----
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

    if "Fuel_Type_Diesel" in input_data:
        input_data["Fuel_Type_Diesel"] = 1 if fuel_type == "Diesel" else 0

    if "Transmission_Manual" in input_data:
        input_data["Transmission_Manual"] = 1 if transmission == "Manual" else 0

    if "Selling_type_Individual" in input_data:
        input_data["Selling_type_Individual"] = 1

    input_df = pd.DataFrame([input_data])[feature_columns]

    # ---- Predict Button ----
    if st.button("🔮 Predict Car Price"):

        prediction = model.predict(input_df)[0]
        prediction = max(0, prediction)

        # Confidence range (±10%)
        lower = prediction * 0.90
        upper = prediction * 1.10

        st.markdown(
            f"""
            <div class='prediction-card'>
            💰 Estimated Selling Price <br><br>
            ₹ {prediction:.2f} Lakhs
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class='range-card'>
            📊 Estimated Price Range: &nbsp; ₹ {lower:.2f} Lakhs &nbsp; — &nbsp; ₹ {upper:.2f} Lakhs
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        # Resale verdict
        if prediction > 10:
            st.success("⭐ High-value vehicle! Great resale potential.")
        elif prediction > 5:
            st.info("✅ Moderate resale value. Fair market price.")
        else:
            st.warning("⚠️ Lower resale value expected. Consider negotiating.")

        # Quick summary
        st.markdown("#### 📋 Prediction Summary")
        summary_df = pd.DataFrame({
            "Parameter": ["Present Price", "KMs Driven", "Car Age", "Fuel Type", "Transmission", "Owners", "Predicted Price"],
            "Value": [
                f"₹ {present_price} Lakhs",
                f"{driven_kms:,} km",
                f"{car_age} Years",
                fuel_type,
                transmission,
                str(owner),
                f"₹ {prediction:.2f} Lakhs"
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.balloons()

# ==========================================
# TAB 2 — EDA
# ==========================================

with tab2:
    st.subheader("📊 Exploratory Data Analysis")

    if df_raw is None:
        st.warning("⚠️ Dataset file 'car_data.csv' not found. EDA charts unavailable.")
    else:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Preprocess for EDA
            df = df_raw.copy()
            df["Car_Age"] = datetime.now().year - df["Year"]

            col1, col2 = st.columns(2)

            # --- Chart 1: Selling Price Distribution ---
            with col1:
                st.markdown("#### 💰 Selling Price Distribution")
                fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0E1117')
                ax.set_facecolor('#161B22')
                ax.hist(df["Selling_Price"], bins=20, color="#00C6FF", edgecolor="#0072FF", alpha=0.85)
                ax.set_xlabel("Selling Price (Lakhs)", color="white")
                ax.set_ylabel("Count", color="white")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                st.pyplot(fig)
                plt.close()

            # --- Chart 2: Fuel Type vs Avg Price ---
            with col2:
                st.markdown("#### ⛽ Fuel Type vs Average Price")
                fuel_avg = df.groupby("Fuel_Type")["Selling_Price"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0E1117')
                ax.set_facecolor('#161B22')
                colors = ["#00C6FF", "#0072FF", "#00FF88"]
                ax.bar(fuel_avg["Fuel_Type"], fuel_avg["Selling_Price"], color=colors[:len(fuel_avg)])
                ax.set_xlabel("Fuel Type", color="white")
                ax.set_ylabel("Avg Selling Price (Lakhs)", color="white")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                st.pyplot(fig)
                plt.close()

            col3, col4 = st.columns(2)

            # --- Chart 3: Car Age vs Selling Price ---
            with col3:
                st.markdown("#### 📅 Car Age vs Selling Price")
                fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0E1117')
                ax.set_facecolor('#161B22')
                ax.scatter(df["Car_Age"], df["Selling_Price"], alpha=0.6, color="#00C6FF", s=40)
                ax.set_xlabel("Car Age (Years)", color="white")
                ax.set_ylabel("Selling Price (Lakhs)", color="white")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                st.pyplot(fig)
                plt.close()

            # --- Chart 4: Transmission vs Avg Price ---
            with col4:
                st.markdown("#### ⚙️ Transmission vs Average Price")
                trans_avg = df.groupby("Transmission")["Selling_Price"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0E1117')
                ax.set_facecolor('#161B22')
                ax.bar(trans_avg["Transmission"], trans_avg["Selling_Price"], color=["#00C6FF", "#0072FF"])
                ax.set_xlabel("Transmission", color="white")
                ax.set_ylabel("Avg Selling Price (Lakhs)", color="white")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                st.pyplot(fig)
                plt.close()

            # --- Chart 5: Present Price vs Selling Price ---
            st.markdown("#### 💹 Present Price vs Selling Price")
            fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
            ax.set_facecolor('#161B22')
            ax.scatter(df["Present_Price"], df["Selling_Price"], alpha=0.6, color="#00FF88", s=40)
            ax.set_xlabel("Present Price (Lakhs)", color="white")
            ax.set_ylabel("Selling Price (Lakhs)", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            st.pyplot(fig)
            plt.close()

            # Key stats
            st.markdown("#### 📈 Key Statistics")
            stats = df[["Selling_Price", "Present_Price", "Driven_kms", "Car_Age"]].describe().round(2)
            st.dataframe(stats, use_container_width=True)

        except ImportError:
            st.error("matplotlib / seaborn not installed. Run: pip install matplotlib seaborn")

# ==========================================
# TAB 3 — MODEL PERFORMANCE
# ==========================================

with tab3:
    st.subheader("🤖 Model Performance & Insights")

    if df_raw is None:
        st.warning("⚠️ Dataset file 'car_data.csv' not found. Model evaluation unavailable.")
    else:
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            import matplotlib.pyplot as plt

            df = df_raw.copy()
            df["Car_Age"] = datetime.now().year - df["Year"]
            df = pd.get_dummies(df, columns=["Fuel_Type", "Seller_Type", "Transmission"], drop_first=False)
            df = df.drop(columns=["Car_Name", "Year"], errors="ignore")

            X = df.drop(columns=["Selling_Price"])
            y = df["Selling_Price"]

            # Align columns
            X = X.reindex(columns=feature_columns, fill_value=0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Metrics display
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>📈</h2>
                    <h3>{r2*100:.2f}%</h3>
                    <p>R² Score (Test Set)</p>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>🎯</h2>
                    <h3>₹ {mae:.2f}L</h3>
                    <p>Mean Absolute Error</p>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>📉</h2>
                    <h3>₹ {rmse:.2f}L</h3>
                    <p>Root Mean Squared Error</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            col1, col2 = st.columns(2)

            # --- Actual vs Predicted ---
            with col1:
                st.markdown("#### 🎯 Actual vs Predicted Prices")
                fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0E1117')
                ax.set_facecolor('#161B22')
                ax.scatter(y_test, y_pred, alpha=0.6, color="#00C6FF", s=40)
                ax.plot([y_test.min(), y_test.max()],
                        [y_test.min(), y_test.max()],
                        color="#FF4B4B", linewidth=2, linestyle="--", label="Perfect Prediction")
                ax.set_xlabel("Actual Price (Lakhs)", color="white")
                ax.set_ylabel("Predicted Price (Lakhs)", color="white")
                ax.tick_params(colors="white")
                ax.legend(facecolor="#161B22", labelcolor="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                st.pyplot(fig)
                plt.close()

            # --- Feature Importance ---
            with col2:
                st.markdown("#### 🏆 Feature Importance")
                importances = model.feature_importances_
                feat_df = pd.DataFrame({
                    "Feature": feature_columns,
                    "Importance": importances
                }).sort_values("Importance", ascending=True)

                fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0E1117')
                ax.set_facecolor('#161B22')
                colors = ["#00C6FF" if i == len(feat_df)-1 else "#0072FF"
                          for i in range(len(feat_df))]
                ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors)
                ax.set_xlabel("Importance Score", color="white")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                st.pyplot(fig)
                plt.close()

            # Honest note
            st.markdown("""
            <div class='info-box'>
            ℹ️ <b>Note:</b> Metrics above are computed on a held-out <b>test set (20%)</b> —
            not training data — to give an honest estimate of real-world performance.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error computing model metrics: {e}")

# ==========================================
# TAB 4 — DATASET PREVIEW
# ==========================================

with tab4:
    st.subheader("📋 Dataset Preview")

    if df_raw is None:
        st.warning("⚠️ Dataset file 'car_data.csv' not found.")
    else:
        st.markdown(f"**Total Records:** {len(df_raw)} &nbsp;&nbsp; **Columns:** {len(df_raw.columns)}")

        st.dataframe(df_raw, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔢 Data Types")
            st.dataframe(df_raw.dtypes.reset_index().rename(
                columns={"index": "Column", 0: "Type"}),
                use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### ❓ Missing Values")
            missing = df_raw.isnull().sum().reset_index()
            missing.columns = ["Column", "Missing Count"]
            st.dataframe(missing, use_container_width=True, hide_index=True)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("""
<center>
Made with ❤️ using Python, Scikit-Learn & Streamlit &nbsp;|&nbsp; Developer: <b>J S Jishnu</b> &nbsp;|&nbsp; B.Tech CSE-AIDE
</center>
""", unsafe_allow_html=True)
