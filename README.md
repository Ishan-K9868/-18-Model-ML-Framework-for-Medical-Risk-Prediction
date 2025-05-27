# Diabetes Prediction Project

A comprehensive machine learning project for diabetes prediction using the Pima Indians Diabetes dataset, featuring exploratory data analysis, data preprocessing, model training, and a web-based prediction application.

## Project Overview

This project demonstrates a complete end-to-end machine learning workflow for healthcare prediction, including:

- Exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering and selection
- Model training and evaluation
- Interactive web application for predictions

## Project Components

This repository contains the following key files:

- `diabetes_prediction_complete.ipynb`: Comprehensive Jupyter notebook with EDA, preprocessing, model training and evaluation
- `simple_prediction_app.py`: Streamlit web application for making diabetes predictions
- `diabetes.csv`: Dataset containing health metrics and diabetes status
- `requirements.txt`: Required Python dependencies

## Dataset Description

The dataset (`diabetes.csv`) includes the following features:
- Gender
- Age
- Hypertension status
- Heart disease status
- Smoking history
- BMI (Body Mass Index)
- HbA1c level
- Blood glucose level
- Diabetes status (target variable)

## Web Application Features

The Streamlit application (`simple_prediction_app.py`) provides:

- User-friendly interface for inputting health metrics
- Real-time diabetes risk prediction
- Visual representation of prediction probability
- Risk level categorization (Low, Moderate, High, Very High)
- Support for advanced features like BMI categorization and glucose-to-insulin ratio

## Getting Started

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook for analysis:
   ```bash
   jupyter notebook diabetes_prediction_complete.ipynb
   ```
4. Launch the prediction application:
   ```bash
   streamlit run simple_prediction_app.py
   ```

## Model Details

The project trains various machine learning models to predict diabetes risk, including:
- Decision Tree classifiers
- Ensemble methods
- Neural networks

The final model is selected based on performance metrics including accuracy, precision, recall, and F1-score.

## Dependencies

Key dependencies include:
- pandas (≥1.3.0)
- numpy (≥1.19.0)
- matplotlib (≥3.4.0)
- seaborn (≥0.11.0)
- scikit-learn (≥1.0)
- xgboost (≥1.6.0)
- catboost (≥1.0.6)
- shap (≥0.39.0)
- tensorflow (≥2.6.0)
- streamlit (≥1.30.0)

See `requirements.txt` for the complete list of dependencies.

## Future Enhancements

- Model explainability improvements using SHAP values
- Integration with electronic health records
- Mobile application development
- Support for additional health metrics and risk factors
- Time-series analysis for monitoring diabetes risk over time

## Note on Usage

This application is designed for educational and research purposes only. It should not replace professional medical advice, diagnosis, or treatment.
