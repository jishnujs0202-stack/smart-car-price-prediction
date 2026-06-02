# 🚗 Smart Car Price Prediction System

## Overview

The Smart Car Price Prediction System is a Machine Learning project that predicts the resale value of used cars based on various features such as kilometers driven, fuel type, transmission type, ownership history, seating capacity, and car age.

The project uses a Random Forest Regression model to provide accurate price predictions and includes data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, and deployment through a Streamlit web application.

---

## Features

* 📊 Data Cleaning and Preprocessing
* 🔧 Feature Engineering (Car Age Calculation)
* 📈 Exploratory Data Analysis (EDA)
* 🤖 Random Forest Regression Model
* 📉 Feature Importance Analysis
* 📋 Model Evaluation using R² Score and MAE
* 🌐 Streamlit Web Application
* ⚡ Real-Time Car Price Prediction

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

## Dataset

The dataset contains information about used cars including:

* Kilometers Driven
* Fuel Type
* Seller Type
* Transmission Type
* Owner History
* Seating Capacity
* Manufacturing Year
* Selling Price

Feature engineering was performed to create the **Car Age** attribute, which significantly improves prediction performance.

---

## Project Workflow

### 1. Data Collection

Loaded the used car dataset and examined the data structure.

### 2. Data Cleaning

* Removed unnecessary columns
* Handled inconsistencies
* Checked for missing values
* Removed duplicate records

### 3. Feature Engineering

Created a new feature:

```python
Car Age = Current Year - Manufacturing Year
```

### 4. Exploratory Data Analysis (EDA)

Performed visual analysis to understand:

* Price distribution
* Fuel type trends
* Transmission effects
* Correlation between features

### 5. Model Training

Trained a **Random Forest Regressor** using engineered features and optimized parameters.

### 6. Model Evaluation

Evaluated model performance using:

* R² Score
* Mean Absolute Error (MAE)

### 7. Deployment

Built and deployed a Streamlit web application for real-time price prediction.

---

## Model Performance

### Algorithm Used

**Random Forest Regressor**

### Evaluation Metrics

| Metric                    | Value      |
| ------------------------- | ---------- |
| R² Score                  | 0.9697     |
| Mean Absolute Error (MAE) | ₹69,612.87 |

The model explains approximately **96.97%** of the variance in car prices, demonstrating strong predictive performance.

---

## Live Demo

🚀 Try the deployed application:

**https://smart-car-price-prediction-hv4ymrvjvqksfeo37gvjsv.streamlit.app/**

Users can enter vehicle details and instantly receive a predicted resale value.

---

## Project Structure

```text
Car_Price_Prediction/
│
├── app.py
├── train_model.py
├── feature_columns.pkl
├── car_price_model.pkl
├── requirements.txt
├── README.md
├── dataset/
│   └── car_data.csv
│
└── notebooks/
    └── EDA.ipynb
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jishnujs0202-stack/smart-car-price-prediction.git
```

Move into the project directory:

```bash
cd smart-car-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will start locally and open in your browser.

---

## Future Improvements

* XGBoost Integration
* Hyperparameter Optimization
* Advanced Feature Engineering
* Interactive Analytics Dashboard
* Cloud Deployment Enhancements
* Car Recommendation System
* Vehicle Market Trend Analysis

---

## Author

### J S Jishnu

B.Tech – Computer Science & Engineering (Artificial Intelligence and Data Engineering)

GitHub: https://github.com/jishnujs0202-stack

---

## License

This project is developed for educational and learning purposes.
