# Crime Safety Prediction System

## Overview
The Crime Safety Prediction System is a data-driven web application that analyzes historical district-wise crime data and estimates future crime trends. The system combines time-based regression forecasting with unsupervised clustering to classify districts into safety zones: Safe, Moderate, or Unsafe. The project is intended for academic and analytical purposes, with a focus on explainable logic rather than real-world law enforcement deployment.

## Objectives
- Forecast future crime counts using historical trends
- Identify latent crime composition patterns using clustering
- Classify districts into safety zones using a hybrid decision strategy
- Provide interactive and interpretable visualizations

## Technology Stack
Backend:
- Python
- Flask
- Custom Linear Regression
- Custom K-Means Clustering

Frontend:
- HTML
- CSS
- JavaScript
- Chart.js

Data:
- CSV-based structured crime dataset (district-wise, year-wise)

## Methodology
The system follows a multi-stage pipeline. First, for each crime category, a custom linear regression model is trained using historical yearly data to capture temporal trends. These models extrapolate future crime counts while ensuring non-negative predictions. Second, a custom K-Means clustering algorithm (k = 3) is trained on scaled crime feature vectors to identify underlying crime composition patterns across districts. Clustering is not directly mapped to safety labels; instead, it provides contextual pattern information. Finally, a hybrid decision logic combines cluster centroid intensity with total predicted crime count to classify districts into Safe, Moderate, or Unsafe zones.

## Features
- District and year-based future crime prediction
- Crime-category-wise prediction table
- Overall crime distribution bar chart
- Crime-type trend visualization (2001–2026)
- Dynamic state–district selection using AJAX
- Pattern-aware safety zone classification

## Dataset
The dataset consists of structured government crime statistics stored in CSV format. Each record contains State/UT, District, Year, and multiple IPC-based crime categories. The dataset is assumed to be pre-cleaned and consistently formatted.

## Project Structure
├── app.py
├── train_model.py
├── evaluate_system.py
├── custom_linear.py
├── custom_kmeans.py
├── custom_scaler.py
├── dataset/
│   └── crime_data.csv
├── model/
│   ├── custom_kmeans.joblib
│   └── custom_scaler.joblib
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── README.md

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Clone the Repository
git clone https://github.com/E-RayZe/crime-safety-prediction
cd crime-safety-prediction

### Install Dependencies
pip install flask pandas numpy joblib

(Optional)
pip install -r requirements.txt

### Train Models (One-Time Setup)
Before running the application, train the clustering and scaling models:
python train_model.py

This will generate the required model files inside the `model/` directory.

### Run the Application
python app.py

Open your browser and navigate to:
http://127.0.0.1:5000/

## Evaluation
The system supports time-based evaluation using historical holdout years. Evaluation metrics include regression error measures such as MAE and RMSE for individual crime categories, as well as safety-zone classification accuracy and confusion matrix analysis. The evaluation focuses on trend consistency rather than real-world certainty.

## Limitations
- Linear regression assumes steady historical trends and cannot capture abrupt changes
- Clustering is relative to historical data distribution
- External socio-economic and demographic factors are not modeled
- Safety zone outputs are indicative and not authoritative

## Intended Use
This project is suitable for academic submissions, final-year engineering or data science projects, and demonstrations of hybrid machine learning systems. It is not intended for real-world law enforcement or policy deployment.

## Author
Shashank, Baljigeshwar
Crime Safety Prediction System
