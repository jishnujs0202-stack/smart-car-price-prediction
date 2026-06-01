import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# ==========================================
# LOAD DATASET
# ==========================================

DATA_PATH = "dataset/car_data.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)


# ==========================================
# CLEAN NUMERIC COLUMNS
# ==========================================

def extract_number(value):
    value = str(value)
    result = pd.Series(value).str.extract(r"(\d+\.?\d*)")[0]
    return float(result.iloc[0]) if pd.notna(result.iloc[0]) else np.nan


df["mileage"] = df["mileage"].apply(extract_number)
df["engine"] = df["engine"].apply(extract_number)
df["max_power"] = df["max_power"].apply(extract_number)

df["mileage"] = df["mileage"].fillna(df["mileage"].median())
df["engine"] = df["engine"].fillna(df["engine"].median())
df["max_power"] = df["max_power"].fillna(df["max_power"].median())
df["seats"] = df["seats"].fillna(df["seats"].median())


# ==========================================
# FEATURE ENGINEERING
# ==========================================

CURRENT_YEAR = datetime.now().year

df["Car_Age"] = CURRENT_YEAR - df["year"]


# ==========================================
# SELECT FEATURES
# ==========================================

features = [
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "Car_Age"
]

target = "selling_price"

X = df[features]
y = df[target]


# ==========================================
# ONE-HOT ENCODING
# ==========================================

X = pd.get_dummies(
    X,
    columns=[
        "fuel",
        "seller_type",
        "transmission",
        "owner"
    ],
    drop_first=False
)

feature_columns = X.columns.tolist()

print("\nModel Features:")
print(feature_columns)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# RANDOM FOREST MODEL
# Smaller model to keep pkl below GitHub 100MB
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)


# ==========================================
# EVALUATION
# ==========================================

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== MODEL PERFORMANCE =====")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : ₹ {mae:,.2f}")
print(f"RMSE     : ₹ {rmse:,.2f}")


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest.pkl", compress=3)
joblib.dump(feature_columns, "feature_columns.pkl")

size_mb = os.path.getsize("models/random_forest.pkl") / (1024 * 1024)

print("\nModel saved successfully!")
print(f"Model file size: {size_mb:.2f} MB")