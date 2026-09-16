# Freight Rate Prediction using XGBoost

An end-to-end machine learning pipeline for predicting freight **posted rates** using historical shipment data. This project was developed for a freight rate prediction technical assessment and includes data preprocessing, feature engineering, hyperparameter optimization, model training, and automated prediction generation.

## Project Overview

The objective of this project is to accurately predict the **posted freight rate** of shipments using operational, geographical, and market-related features. The final solution generates prediction files for both a hidden validation dataset and a fixed December 2025 pricing scenario.

| | |
|---|---|
| **Problem Type** | Supervised Regression |
| **Model** | XGBoost Regressor |
| **Dataset** | 48,000 historical shipment records |

---

## Model Performance

| Metric | Score |
| --- | ---: |
| **R² Score** | **0.8613** |
| **RMSE** | **544.68** |
| **MAE** | **126.32** |

The optimized XGBoost model explains **86.13%** of the variance in freight pricing while maintaining an average prediction error of approximately **$126** on unseen validation data.

---

## Project Structure

```text
freight-rate-prediction/
│
├── data/
│   ├── train-test.csv
│   ├── validation.csv
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   └── candidate_december.png
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── evaluate.py
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
│   └── __init__.py
│
├── main.py
├── score.py
├── requirements.txt
├── validation_predictions.csv
├── december_predictions.csv
├── freight-rate-ml-assessment.pdf
└── README.md
```

---

## Key Features

- Missing value handling using median imputation
- Negative weight correction
- Temporal feature extraction (Year, Month, Day, Weekday)
- Haversine distance calculation
- Weight-per-mile feature engineering
- Market-distance interaction feature
- One-Hot Encoding for categorical variables
- XGBoost Regression model
- Hyperparameter tuning using RandomizedSearchCV
- 5-Fold Cross Validation
- Automated generation of submission-ready prediction files

---

## Technologies Used

| Category | Technology |
| --- | --- |
| Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost |
| Model Selection | Scikit-learn |
| Model Persistence | Joblib |

---

## Installation

Clone the repository and install the required dependencies.

```bash
git clone <repository-url>
cd freight-rate-prediction
pip install -r requirements.txt
```

---

## Training the Model

Run the complete training pipeline.

```bash
python main.py
```

This command will:

1. Load and preprocess the training dataset.
2. Perform feature engineering.
3. Split data using an 80:20 train-validation strategy.
4. Optimize XGBoost using RandomizedSearchCV (5-fold CV).
5. Save the trained model.
6. Generate validation and December prediction files.

---

## Evaluate Predictions

Run the provided scoring utility.

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv --output-dir reports
```

This validates the generated prediction files and creates the required **candidate_december.png** visualization.

---

## Data Validation Strategy

The original **train-test.csv** dataset was divided into an **80:20 train-validation split** using `random_state = 42` to ensure reproducibility. Hyperparameter optimization was performed using **RandomizedSearchCV** with **12 candidate configurations** and **5-fold cross-validation**. The final model was selected based on the lowest cross-validation RMSE.

---

## Output Files

| File | Description |
| --- | --- |
| `best_model.pkl` | Trained XGBoost model |
| `validation_predictions.csv` | Final validation predictions |
| `december_predictions.csv` | December 2025 predictions |
| `candidate_december.png` | Fixed December prediction chart |

---

## Author

**Sakthivel N**

Machine Learning & AI Engineer
