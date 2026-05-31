# 🚗 Smart Car Price Prediction System

## Overview

Smart Car Price Prediction System is a Machine Learning project that predicts the resale value of used cars based on features such as present price, fuel type, transmission type, kilometers driven, ownership history, and car age.

The project uses Random Forest Regression for accurate price prediction and includes data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, and deployment.

---

## Features

* Data Cleaning and Preprocessing
* Feature Engineering (Car Age Calculation)
* Exploratory Data Analysis (EDA)
* Random Forest Regression Model
* Feature Importance Analysis
* Model Evaluation using R² Score, MAE, and RMSE
* Streamlit Web Application for Real-Time Prediction

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## Project Structure

```text
Car_Price_Prediction/
│
├── dataset/
│   └── car_data.csv
│
├── models/
│   └── random_forest.pkl
│
├── app.py
├── train_model.py
├── feature_columns.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Performance

* Algorithm: Random Forest Regressor
* Evaluation Metrics:

  * R² Score
  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python train_model.py
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Future Improvements

* XGBoost Integration
* Hyperparameter Tuning
* Interactive Visual Analytics Dashboard
* Cloud Deployment
* Car Recommendation System

---

## Author

Jishnu S
B.Tech CSE-AIDE
