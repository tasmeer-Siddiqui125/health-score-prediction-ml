# Health Score Prediction Using Machine Learning

##  Project Overview

This project develops a machine learning regression model to predict **average health score** using demographic, lifestyle, physiological, and health-related features.

The project follows an end-to-end machine learning lifecycle, starting from exploratory data analysis and preprocessing and continuing through model training, evaluation, hyperparameter tuning, cross-validation, and feature-importance analysis.

---

##  Objective

The main objective is to predict the health score of a patient based on available health, lifestyle, demographic, and risk-related information.

This project also aims to:

- Understand patterns and relationships within the dataset
- Identify important predictors of health score
- Compare different regression algorithms
- Improve model performance through hyperparameter tuning
- Select and evaluate the best-performing regression model

---

##  Dataset Features

The dataset contains demographic, lifestyle, physiological, and health-related variables, including:

- Age
- BMI
- Blood Pressure
- Cholesterol
- Glucose
- Heart Rate
- Oxygen Level
- Sleep Hours
- Exercise Hours
- Stress Level
- Alcohol Consumption
- Monthly Analyses
- Biomarker Abnormal Rate
- Average Metabolic Risk
- Average Cardiovascular Risk
- Gender
- Region
- Smoking Status
- Year
- Month
- Cumulative Analyses

### Target Variable

**Average Health Score**

---

##  Exploratory Data Analysis

The following exploratory analyses were performed:

- Dataset overview
- Missing-value analysis
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Correlation heatmap
- Distribution and skewness analysis
- Categorical feature analysis
- Feature vs. target analysis
- Time-based trend analysis

### Key EDA Findings

- Several numerical features showed positive skewness.
- `monthly_analyses`, `alcohol_consumption`, and `exercise_hours` showed relatively high positive skewness.
- Correlation analysis was used to understand relationships between numerical variables.
- Health score remained relatively stable over time and across monthly analysis, with a noticeable increase in cumulative analysis during the second half of 2022.
- `exercise_hours` showed an important relationship with health score.

---

##  Data Preprocessing

The following preprocessing steps were performed:

### Missing Values

Numerical missing values were handled using **median imputation**.

Categorical missing values were handled using **most-frequent-value imputation**.

### Categorical Encoding

Categorical variables such as:

- Gender
- Region
- Smoking Status

were converted into numerical representations using **One-Hot Encoding**.

### Skewness Transformation

Numerical feature distributions were examined both numerically and visually.

Features with significant positive skewness were considered for transformation. **Yeo-Johnson transformation** was used because it can handle both positive and negative values and does not require strictly positive input values.

### Feature Scaling

`StandardScaler` was applied to standardize numerical features.

---

##  Machine Learning Models

The following regression models were evaluated:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Decision Tree Regressor
6. Random Forest Regressor
7. Gradient Boosting Regressor
8. XGBoost Regressor

Gradient Boosting and XGBoost were then further improved using hyperparameter tuning.

---

##  Model Evaluation

The models were evaluated using:

- **MAE (Mean Absolute Error)** — lower is better
- **RMSE (Root Mean Squared Error)** — lower is better
- **R² Score** — higher is better

### Final Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 2.4884 | 3.1744 | 0.5856 |
| Ridge Regression | 2.4884 | 3.1744 | 0.5856 |
| Lasso Regression | 2.5075 | 3.1830 | 0.5833 |
| ElasticNet | 2.5097 | 3.1845 | 0.5829 |
| Decision Tree | 3.6937 | 4.7330 | 0.0786 |
| Random Forest | 2.5442 | 3.2200 | 0.5736 |
| Gradient Boosting | 2.4813 | 3.1565 | 0.5902 |
| XGBoost | 2.4877 | 3.1668 | 0.5875 |
| Tuned Gradient Boosting | 2.4529 | 3.1355 | 0.5956 |
| **Tuned XGBoost** | **2.4614** | **3.1349** | **0.5958** |

---

##  Hyperparameter Tuning

Gradient Boosting and XGBoost were tuned using `GridSearchCV` with 5-fold cross-validation.

### Best Gradient Boosting Parameters

```text
learning_rate = 0.1
max_depth = 2
n_estimators = 200