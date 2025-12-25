# IMDb Score Prediction 🎬

## Project Overview
This project focuses on predicting **IMDb movie scores** using machine learning regression techniques.  
The goal is to analyze movie metadata, perform exploratory data analysis (EDA), and build predictive models to estimate IMDb ratings based on various features.

The project follows a **progressive modeling approach**, starting from simple models and moving toward more advanced ensemble methods to improve performance.

---

## 🎯 Problem Statement
IMDb score prediction is a **regression problem**, where the target variable is a continuous value representing a movie’s rating.

**Objective:**  
Build a robust regression model that accurately predicts IMDb scores while avoiding underfitting and overfitting.

---

## 🧠 Dataset Description
The dataset contains metadata related to movies, such as:
- Movie duration
- Budget & gross revenue
- Number of votes
- Director & actor-related features
- Genres
- Other numerical and categorical attributes

**Target Variable:**  
- `imdb_score`

---

## 🔍 Exploratory Data Analysis (EDA)
Key EDA steps performed:
- Distribution analysis of IMDb scores
- Correlation analysis using heatmaps
- Detection of skewness and outliers
- Feature importance analysis
- Understanding relationships between numerical features and IMDb score

---

## ⚙️ Models Implemented
Multiple regression models were trained and evaluated to compare performance:

| Model | Purpose |
|------|--------|
| **Linear Regression** | Baseline model to understand linear relationships |
| **Decision Tree Regressor** | Captures non-linear patterns but prone to overfitting |
| **Random Forest Regressor** | Reduces overfitting using ensemble learning |
| **XGBoost Regressor** | Gradient boosting for better generalization and performance |
| **Artificial Neural Network (ANN)** | Tested but underperformed due to limited data |

---

## 🚀 Why XGBoost?
XGBoost was selected as the **final model** because:
- It handles **non-linear relationships** effectively
- Strong **regularization** reduces overfitting
- Performs well on structured/tabular data
- Achieved the **best R² score** among all models
- More stable compared to Decision Trees

ANN underperformed due to **insufficient data**, leading to underfitting.

---

## 📊 Model Evaluation Metrics
The models were evaluated using:
- **R² Score**
- **Root Mean Squared Error (RMSE)**

XGBoost consistently outperformed other models on the test dataset.

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries & Tools:**
  - NumPy
  - Pandas
  - Matplotlib & Seaborn
  - Scikit-learn
  - XGBoost
  - TensorFlow





