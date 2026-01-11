# Data Science Salary Prediction Model

**Machine Learning Model for Predicting Data Science Salaries using XGBoost**
---

## Project Overview

This project develops a machine learning model to predict salaries for data science professionals based on various factors such as job title, experience level, company size, and geographic location. The model helps both job seekers understand salary expectations and employers set competitive compensation packages.

**Google Colab:** [View Notebook](https://colab.research.google.com/drive/1ojXN6RJi1uwWXuzWxSXwYGQEguH3XDYG?usp=sharing)

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Records** | 5,736 |
| **Features** | 11 |
| **Job Titles** | 132 unique roles |
| **Countries** | 75 |
| **Time Period** | 2020 - 2024 |
| **Data Source** | Kaggle (Sourav Banerjee) |

### Salary Distribution (USD)

| Statistic | Value |
|-----------|-------|
| Minimum | $15,000 |
| Maximum | $750,000 |
| Mean | $144,264 |
| Median | $136,772 |

---

## Features

The model uses the following input features:

| Feature | Description | Type |
|---------|-------------|------|
| Job Title | Role designation (Data Scientist, ML Engineer, etc.) | Categorical |
| Employment Type | Full-Time, Part-Time, Contract, Freelance | Categorical |
| Experience Level | Entry, Mid, Senior, Executive | Categorical |
| Expertise Level | Junior, Intermediate, Expert, Director | Categorical |
| Company Size | Small, Medium, Large | Categorical |
| Company Location | Geographic region of the company | Categorical |
| Employee Residence | Employee's country of residence | Categorical |
| Year | Year of the salary data | Numerical |

**Target Variable:** Salary in USD

---

## Data Distribution

### Employment Type

| Type | Count | Percentage |
|------|-------|------------|
| Full-Time | 5,690 | 99.2% |
| Contract | 19 | 0.3% |
| Part-Time | 15 | 0.3% |
| Freelance | 12 | 0.2% |

### Experience Level

| Level | Count | Avg. Salary (USD) |
|-------|-------|-------------------|
| Senior | 3,530 | $161,576 |
| Mid | 1,455 | $116,884 |
| Entry | 524 | $84,356 |
| Executive | 227 | $188,840 |

### Company Size

| Size | Count | Avg. Salary (USD) |
|------|-------|-------------------|
| Medium | 5,016 | $148,951 |
| Large | 550 | $119,009 |
| Small | 170 | $87,687 |

### Top 10 Job Titles

| Rank | Job Title | Count |
|------|-----------|-------|
| 1 | Data Engineer | 1,097 |
| 2 | Data Scientist | 1,051 |
| 3 | Data Analyst | 772 |
| 4 | Machine Learning Engineer | 542 |
| 5 | Analytics Engineer | 218 |
| 6 | Research Scientist | 177 |
| 7 | Data Architect | 149 |
| 8 | Research Engineer | 113 |
| 9 | ML Engineer | 102 |
| 10 | Data Science Manager | 83 |

### Top 10 Countries

| Rank | Country | Count |
|------|---------|-------|
| 1 | United States | 4,564 |
| 2 | United Kingdom | 377 |
| 3 | Canada | 218 |
| 4 | Germany | 75 |
| 5 | Spain | 60 |
| 6 | India | 54 |
| 7 | France | 49 |
| 8 | Australia | 28 |
| 9 | Portugal | 26 |
| 10 | Netherlands | 21 |

---

## Model Comparison

Three machine learning models were evaluated:

| Model | MAE | RMSE | R-squared |
|-------|-----|------|-----------|
| Decision Tree | ~1,750 | ~20,000 | ~0.40 |
| Random Forest | ~1,250 | ~15,000 | ~0.60 |
| **XGBoost** | **Lowest** | **Lowest** | **Highest** |

**Selected Model: XGBoost** - Chosen for its superior performance across all metrics.

### Why XGBoost?

XGBoost (Extreme Gradient Boosting) was selected because:

1. Lowest Mean Absolute Error (MAE)
2. Lowest Root Mean Squared Error (RMSE)
3. Highest R-squared value
4. Efficient handling of missing values
5. Built-in regularization to prevent overfitting
6. Parallel tree boosting for faster training

---

## Project Pipeline

```
1. Data Loading
   └── Import dataset from Kaggle

2. Data Preprocessing
   ├── Identify categorical vs numerical columns
   ├── Handle missing values
   └── One-hot encoding for categorical features

3. Feature Engineering
   ├── Separate features (X) and target (y)
   └── Train-test split (80/20)

4. Data Scaling
   └── StandardScaler for numerical features

5. Model Training
   ├── Decision Tree Regressor
   ├── Random Forest Regressor
   └── XGBoost Regressor

6. Model Evaluation
   ├── MAE (Mean Absolute Error)
   ├── RMSE (Root Mean Squared Error)
   └── R-squared Score

7. Hyperparameter Tuning
   └── Optimize XGBoost parameters

8. Prediction Interface
   └── Interactive salary prediction
```

---

## Installation

### Prerequisites

- Python 3.x
- pip package manager

### Required Libraries

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

### Running the Model

1. Clone the repository:
```bash
git clone https://github.com/yourusername/salary-prediction-model.git
cd salary-prediction-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook Salary_Pred_Model.ipynb
```

Or use Google Colab directly via the link provided above.

---

## Usage

### Interactive Prediction

The model provides an interactive interface for salary prediction:

```python
# Example Input
Job Title: Data Scientist
Employment Type: Full-Time
Experience Level: Senior
Expertise Level: Expert
Company Size: Large
Company Location: United States
Year: 2024

# Output
Predicted Salary: $XXX,XXX.XX
```

### Sample Code

```python
# Load and preprocess user input
user_data = pd.DataFrame({
    'Job Title': ['Data Scientist'],
    'Employment Type': ['Full-Time'],
    'Experience Level': ['Senior'],
    'Expertise Level': ['Expert'],
    'Company Size': ['Large'],
    'Company Location': ['United States'],
    'Year': [2024]
})

# Preprocess and predict
user_data_processed = preprocess_user_input(user_data, encoder, scaler, full_feature_list)
predicted_salary = predict_salary(xgboost_model, user_data_processed.values)

print(f"Predicted Salary: ${predicted_salary:,.2f}")
```

---

## Visualizations

The project includes several visualizations:

1. **Scatter Plot** - Actual vs Predicted Salaries
2. **Histogram** - Error Distribution (Residuals)
3. **Bar Chart** - Model Performance Comparison (MAE, RMSE, R-squared)
4. **Feature Importance** - Top contributing features

---

## Repository Structure

```
salary-prediction-model/
│
├── README.md                      # Project documentation
├── Salary_Pred_Model.ipynb        # Jupyter notebook with full implementation
├── Salary_Pred_model.pdf          # Detailed project report
├── Salaries.csv                   # Dataset
├── requirements.txt               # Python dependencies
│
└── docs/
    └── images/                    # Visualization screenshots
        ├── scatter_plot.png
        ├── error_distribution.png
        ├── model_comparison.png
        └── feature_importance.png
```

---

## Key Findings

1. **Experience Matters Most**: Executive and Senior-level positions command significantly higher salaries ($160K-$190K average).

2. **Company Size Impact**: Medium-sized companies offer the highest average salaries ($149K), followed by Large ($119K) and Small ($88K).

3. **Geographic Concentration**: 79.5% of the data comes from the United States, reflecting the dominant tech market.

4. **Role Hierarchy**: Data Engineers and Data Scientists are the most common roles, with ML Engineers commanding premium salaries.

5. **Full-Time Dominance**: 99.2% of positions are full-time, indicating stable employment in the field.

---

## Limitations

- Dataset is heavily skewed towards US-based positions
- Limited representation of Part-Time, Contract, and Freelance roles
- Salary data may not reflect total compensation (bonuses, equity, benefits)
- Model accuracy depends on the quality and recency of training data

---

## Future Improvements

- Incorporate additional features (education, certifications, skills)
- Add more data from underrepresented regions
- Implement real-time data updates
- Deploy as a web application
- Include confidence intervals for predictions

---

## Author

**Aryan Bansal**   
University of Technology Sydney

---

## References

1. NVIDIA Data Science Glossary - [What is XGBoost?](https://www.nvidia.com/en-au/glossary/xgboost/)
2. GeeksforGeeks - [XGBoost](https://www.geeksforgeeks.org/xgboost/)
3. Kaggle Dataset - [Data Science Salaries](https://www.kaggle.com/datasets/iamsouravbanerjee/data-science-salaries-2023/data)
4. XGBoost Documentation - [Python Package Introduction](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)

---


---

*Advanced Data Analytics Algorithm - Machine Learning | Spring 2024*
