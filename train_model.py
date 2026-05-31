# ==========================================
# SMART CAR PRICE PREDICTION SYSTEM
# ==========================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/car_data.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================
# FEATURE ENGINEERING
# ==========================================

CURRENT_YEAR = 2026

df["Car_Age"] = CURRENT_YEAR - df["Year"]

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(8,5))
sns.histplot(df["Selling_Price"], kde=True)
plt.title("Selling Price Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="Car_Age",
    y="Selling_Price"
)
plt.title("Car Age vs Selling Price")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Fuel_Type",
    y="Selling_Price"
)
plt.title("Fuel Type vs Selling Price")
plt.show()

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(8,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# PREPROCESSING
# ==========================================

df.drop(["Car_Name", "Year"], axis=1, inplace=True)

df = pd.get_dummies(
    df,
    drop_first=True
)

# ==========================================
# SAVE FEATURE COLUMNS
# ==========================================

X = df.drop("Selling_Price", axis=1)

feature_columns = X.columns

joblib.dump(
    feature_columns,
    "feature_columns.pkl"
)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

r2 = r2_score(y_test, predictions)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

print("\n========================")
print("MODEL PERFORMANCE")
print("========================")

print("R2 Score :", round(r2,4))
print("MAE      :", round(mae,4))
print("RMSE     :", round(rmse,4))

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features")
print(importance.head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.show()

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "random_forest.pkl"
)

print("\nModel Saved Successfully")
print("random_forest.pkl")
print("feature_columns.pkl")