# Telco Customer Churn Prediction — Machine Learning & Retention Analysis

---

## Overview

Customer churn — when subscribers cancel or stop using a service — is one of the most damaging and preventable problems in subscription-based businesses. Acquiring a new customer costs five to seven times more than retaining an existing one, which means every churned customer represents a compounded loss.

This project analyses 7,043 telecom customer records to identify the behavioural and contractual patterns that precede churn, and builds three machine learning models to predict which customers are at risk before they leave. The best model — XGBoost — achieved an AUC-ROC of 0.84, meaning it correctly distinguishes between loyal and at-risk customers 84% of the time.

The output is not just a model. It is a set of specific, data-backed recommendations that a retention team can act on immediately.

---

## Business Insights

After analysing the full dataset, four patterns emerged that explain the majority of churn behaviour.

**The Contract Trap**
Customers on month-to-month contracts churn at dramatically higher rates than those on one or two-year plans. The flexibility that attracts these customers is the same flexibility that makes it easy for them to leave. Converting even a fraction of this segment to annual contracts would have a measurable impact on retention.

**The Fiber Optic Paradox**
Fiber optic internet customers — the highest-value subscribers — churn more than DSL or no-internet customers. This is counterintuitive. It suggests that price-to-value perception is broken in this segment, or that service reliability issues are driving dissatisfaction among the most engaged users.

**Payment Friction**
Customers paying by electronic check churn significantly more than those on automatic bank transfers or credit cards. Manual payment methods introduce friction and missed payments, both of which are associated with disengagement. This is one of the easiest levers to pull — incentivising a switch to auto-pay costs very little.

**The Newbie Phase**
Churn is heavily concentrated in the first twelve months. Customers who survive past year one become substantially more loyal. The first year is the critical retention window — most of the business risk lives there.

---

## Visual Analysis

**Churn Distribution**

![Churn Distribution](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/churn_distribution.png)

*26.54% of customers in the dataset have churned — a meaningful rate that justifies a dedicated predictive modelling effort.*

**Key Categorical Drivers**

![Categorical Features by Churn](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/categorical_features.png)

*Contract type, internet service type, and payment method are the three categorical variables with the strongest relationship to churn.*

**Tenure and Numerical Features**

![Numerical Features by Churn](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/numerical_features.png)

*Short-tenure customers and those with higher monthly charges show disproportionately higher churn rates.*

**Churn by Tenure Group**

![Tenure Group Churn Rate](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/tenure_group_churn.png)

*Customers in their first 12 months churn at the highest rate. The risk drops sharply after the first year and continues declining with tenure.*

---

## Model Development

Three models were built and evaluated. All models used an 80/20 stratified train-test split to preserve the class distribution across both sets.

**Preprocessing pipeline:**
- Missing values in TotalCharges filled with column median
- Binary categorical columns (Yes/No, gender) encoded as 0/1
- Multi-category columns one-hot encoded
- Numerical features standardised using StandardScaler
- A total_services feature was engineered to capture how many active services each customer holds

**Hyperparameter tuning:**
Both Random Forest and XGBoost were tuned using 5-fold cross-validated GridSearchCV optimising for F1 score on the training set.

---

## Model Results

| Model | AUC-ROC | F1 Score (Churn class) |
|---|---|---|
| Logistic Regression | Baseline | — |
| Random Forest | — | — |
| XGBoost | **0.840** | **0.58** |

**XGBoost was selected as the final model** based on its superior AUC-ROC score.

**ROC Curves — All Three Models**

![ROC Curves](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/roc_curves.png)

**Confusion Matrices**

![Confusion Matrices](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/confusion_matrices.png)

---

## Feature Importance

![XGBoost Feature Importance](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/feature_importance.png)

The five features with the highest predictive weight in the XGBoost model:

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | Internet Service — Fiber Optic | 0.193 |
| 2 | Contract — One Year | 0.191 |
| 3 | Contract — Two Year | 0.190 |
| 4 | Internet Service — No Internet | 0.081 |
| 5 | Payment Method — Electronic Check | 0.073 |

Contract type and internet service together account for over 65% of the model's predictive power. These are not marginal factors — they are the primary drivers of churn.

---

## Correlation Analysis

![Correlation Matrix](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/correlation_matrix.png)

*Tenure shows the strongest negative correlation with churn. Monthly charges show a slight positive correlation. Total charges correlates strongly with tenure, as expected.*

---

## Recommendations

**Incentivise long-term contracts**
Offer a meaningful discount — not a token gesture — to month-to-month customers willing to commit to a one-year plan. Contract type is the second and third most important feature in the model. Moving customers off month-to-month is the single highest-impact retention lever available.

**Investigate the fiber optic experience**
Fiber optic internet is the top predictive feature for churn. This requires operational investigation, not just a marketing response. Is there a reliability issue? A pricing problem? A gap between what was promised and what is delivered? The data signals the problem — the business needs to find the root cause.

**Convert electronic check users to auto-pay**
Offer a one-time bill credit to customers who switch from electronic checks to automatic bank transfer or credit card billing. The cost of the incentive is small relative to the cost of losing the customer.

**Build a first-year engagement programme**
Churn is disproportionately concentrated in the first twelve months. A structured onboarding and check-in programme — proactive outreach at 30, 90, and 180 days — directly targets the period of highest risk.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3 |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Visualisation | Matplotlib, Seaborn |
| Model persistence | Joblib |
| Hyperparameter tuning | GridSearchCV (5-fold CV) |
| Version control | Git and GitHub |

---

## Repository Structure
```
Telco_Customer_Churn_Prediction/
├── data/
│   ├── Raw_data.csv                      # Original IBM Telco dataset
│   └── cleaned_data.csv                  # Cleaned and preprocessed dataset
├── images/
│   ├── churn_distribution.png            # Target variable distribution
│   ├── numerical_features.png            # Tenure, charges histograms by churn
│   ├── categorical_features.png          # Contract, internet, payment by churn
│   ├── correlation_matrix.png            # Numerical feature correlations
│   ├── tenure_group_churn.png            # Churn rate by tenure band
│   ├── roc_curves.png                    # ROC curves for all three models
│   ├── confusion_matrices.png            # Confusion matrices for all three models
│   └── feature_importance.png            # XGBoost top feature importances
├── models/
│   ├── logistic_regression_model.pkl     # Saved logistic regression model
│   ├── random_forest_model.pkl           # Saved random forest model
│   └── xgboost_model.pkl                 # Saved XGBoost model (best)
├── churn_analysis.py                     # Full pipeline — EDA, preprocessing, modelling
├── eda_summary.txt                       # EDA findings summary
├── model_insights.txt                    # Model results and feature importances
├── requirements.txt                      # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## Running the Project

**1. Clone the repository**
```bash
git clone https://github.com/najeebullahii/Telco_Customer_Churn_Prediction.git
cd Telco_Customer_Churn_Prediction
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the full pipeline**
```bash
python churn_analysis.py
```

Running this script executes the complete pipeline in sequence — data cleaning, exploratory analysis, feature engineering, model training with hyperparameter tuning, evaluation, and result export. All plots are saved to the `images/` folder and all trained models are saved to the `models/` folder automatically.

---

## Dataset

IBM Telco Customer Churn Dataset — available on Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

7,043 customer records across 21 features including contract type, internet service, payment method, tenure, and monthly charges.

---

## Limitations

- The dataset is synthetically generated by IBM for demonstration purposes and may not perfectly reflect real-world telecom customer behaviour
- Class imbalance (26.54% churn vs 73.46% retained) was not explicitly addressed with techniques such as SMOTE or class weighting — this is a potential improvement
- The model predicts churn probability but does not estimate the financial value of retaining each customer, which would be necessary for prioritising outreach

---

## License

MIT License — free to use and adapt for your own projects.
