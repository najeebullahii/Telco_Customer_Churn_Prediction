# Telco Customer Churn Prediction — Machine Learning & Retention Analysis

---

## What This Project Is About

Telecom companies lose customers every month and most of them never really know why until it's too late. This project tries to change that.

I analysed 7,043 telecom customer records to figure out which customers are most likely to cancel their service — and more importantly, why. The end goal was not just to build a model that predicts churn, but to extract actionable patterns that a retention team could actually use.

The project covers the full pipeline: exploratory data analysis, feature engineering, training and comparing three machine learning models, and translating the results into specific business recommendations. The best model — XGBoost — correctly identifies at-risk customers 84% of the time based on AUC-ROC.

The current churn rate in this dataset is 26.54%. That is roughly one in four customers. For any subscription business, that number demands attention.

---

## What I Found

The analysis kept pointing back to the same four patterns regardless of which angle I looked from.

**Contract type is everything.** Month-to-month customers churn at a rate that is completely out of proportion to the rest of the base. The flexibility that attracts them is the same flexibility that makes it trivially easy to leave. This turned out to be the second and third most important feature in the final model, which tells you how dominant this factor really is.

**Fiber optic customers are unhappy.** This one surprised me. You would expect the highest-tier internet subscribers to be the most satisfied — they are paying for the best product. Instead, they churn more than DSL users and significantly more than customers with no internet service. Something is broken in that experience, whether it is pricing, reliability, or a gap between what was promised and what is actually delivered.

**How people pay predicts whether they stay.** Customers paying by electronic check churn at noticeably higher rates than those on automatic bank transfers or credit cards. Manual payment methods introduce friction, missed payments, and a general sense of lower commitment. This is one of the easiest things to fix.

**The first year is when you lose people.** Churn is heavily concentrated in the 0 to 12 month window. Customers who make it past the first year become dramatically more loyal. If a retention programme exists, it should be almost entirely focused on new customers.

---

## Visual Analysis

**Churn Distribution**

![Churn Distribution](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/churn_distribution.png)

*Just over a quarter of customers have churned. That is high enough to justify a dedicated predictive effort.*

**Key Categorical Drivers**

![Categorical Features by Churn](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/categorical_features.png)

*Contract type, internet service, and payment method — these three variables explain most of what is happening.*

**Tenure and Charges**

![Numerical Features by Churn](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/numerical_features.png)

*Short-tenure customers and those with higher monthly charges are far more likely to churn.*

**Churn by Tenure Group**

![Tenure Group Churn Rate](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/tenure_group_churn.png)

*The first twelve months are the danger zone. After that, the churn rate drops sharply and keeps falling.*

---

## The Models

I built three models and compared them properly rather than just picking the most popular one.

All models used an 80/20 stratified train-test split. Both Random Forest and XGBoost were tuned with 5-fold cross-validated GridSearchCV optimising for F1 score, since a model that ignores the minority class is useless for churn prediction.

Before modelling, the data went through a proper preprocessing pipeline — missing TotalCharges values filled with the column median, binary columns encoded as 0/1, multi-category columns one-hot encoded, and numerical features standardised. I also engineered a total_services feature that counts how many active services each customer holds, which turned out to be a useful signal.

**Results:**

| Model | AUC-ROC | F1 Score (Churn) |
|---|---|---|
| Logistic Regression | Baseline | — |
| Random Forest | — | — |
| XGBoost | **0.840** | **0.58** |

XGBoost won. The ROC curves below show the separation clearly.

**ROC Curves**

![ROC Curves](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/roc_curves.png)

**Confusion Matrices**

![Confusion Matrices](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/confusion_matrices.png)

---

## What the Model Says Matters

![XGBoost Feature Importance](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/feature_importance.png)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Internet Service — Fiber Optic | 0.193 |
| 2 | Contract — One Year | 0.191 |
| 3 | Contract — Two Year | 0.190 |
| 4 | Internet Service — No Internet | 0.081 |
| 5 | Payment Method — Electronic Check | 0.073 |

Contract type and internet service together account for over 65% of the model's predictive power. That is not a subtle finding — those two variables essentially tell you whether a customer is going to stay or leave.

**Correlation Matrix**

![Correlation Matrix](https://raw.githubusercontent.com/najeebullahii/Telco_Customer_Churn_Prediction/main/images/correlation_matrix.png)

*Tenure has the strongest negative relationship with churn. The longer someone has been a customer, the less likely they are to leave — which makes the first-year focus even more important.*

---

## Recommendations

**Move people off month-to-month contracts.** This is the highest-impact action available. A meaningful discount — not a token offer — for switching to a one-year plan directly addresses the top predictive factors in the model. The cost of the discount is small compared to the lifetime value of a retained customer.

**Find out what is wrong with fiber optic.** The model flags this as the single most important churn predictor. A marketing response alone will not fix it. This needs an operational investigation — service reliability data, support ticket analysis, customer satisfaction scores specific to that segment. Something is causing high-value customers to leave at higher rates than everyone else.

**Get electronic check users onto auto-pay.** A one-time bill credit for switching to automatic payment is a cheap intervention that targets a measurable risk factor. It also reduces payment friction and missed payments, which have their own downstream effects.

**Build a structured first-year programme.** Since most churn happens in year one, a proactive outreach sequence — check-ins at 30, 90, and 180 days — directly addresses the highest-risk window. This does not need to be expensive. A well-timed email or call at the right moment can change the trajectory.

---

## How It Was Built

**Data Cleaning**
- TotalCharges had blank strings instead of nulls for new customers — converted and filled with the column median
- Removed duplicate rows and dropped the customerID column
- Saved cleaned dataset to `data/cleaned_data.csv`

**Feature Engineering**
- Created tenure_group bands (0-12, 13-24, 25-36, 37-48, 49-60, 61-72 months)
- Engineered total_services feature counting active subscriptions per customer
- Binary encoded Yes/No columns and gender
- One-hot encoded Contract, InternetService, PaymentMethod, and tenure_group

**Modelling**
- Train/test split: 80/20 stratified on the churn label
- Logistic Regression as interpretable baseline
- Random Forest and XGBoost both tuned with GridSearchCV across key hyperparameters
- All three models saved as pickle files for reuse

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Visualisation | Matplotlib, Seaborn |
| Model persistence | Joblib |
| Tuning | GridSearchCV with 5-fold cross-validation |

---

## Repository Structure
```
Telco_Customer_Churn_Prediction/
├── data/
│   ├── Raw_data.csv                      # Original IBM Telco dataset
│   └── cleaned_data.csv                  # Cleaned and preprocessed dataset
├── images/
│   ├── churn_distribution.png            # Target variable overview
│   ├── numerical_features.png            # Tenure and charges by churn
│   ├── categorical_features.png          # Contract, internet, payment by churn
│   ├── correlation_matrix.png            # Feature correlations
│   ├── tenure_group_churn.png            # Churn rate broken down by tenure band
│   ├── roc_curves.png                    # ROC curves comparing all three models
│   ├── confusion_matrices.png            # Confusion matrices for all three models
│   └── feature_importance.png            # XGBoost top feature importances
├── models/
│   ├── logistic_regression_model.pkl     # Saved logistic regression
│   ├── random_forest_model.pkl           # Saved random forest
│   └── xgboost_model.pkl                 # Saved XGBoost — best performing model
├── churn_analysis.py                     # Full pipeline script
├── eda_summary.txt                       # EDA findings
├── model_insights.txt                    # Model results and feature importances
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Running the Project
```bash
git clone https://github.com/najeebullahii/Telco_Customer_Churn_Prediction.git
cd Telco_Customer_Churn_Prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python churn_analysis.py
```

Running `churn_analysis.py` executes the full pipeline from cleaning through to model evaluation. All plots save automatically to `images/` and all trained models save to `models/`.

---

## Dataset

IBM Telco Customer Churn Dataset — available on Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Limitations

The dataset is synthetically generated by IBM which means it may not capture the full complexity of real-world churn behaviour. Class imbalance was not explicitly addressed with techniques like SMOTE or class weighting, which is a reasonable next step. The model also does not estimate the financial value of each at-risk customer, which would be needed to properly prioritise retention outreach.

---

## License

MIT License — free to use for your own projects.
