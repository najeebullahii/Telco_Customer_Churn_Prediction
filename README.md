# Telco Customer Churn: Why Customers Leave and How to Keep Them

## Project Overview
Losing customers (known as "churn") is one of the biggest challenges for any subscription-based business. This project analyzes a telecommunications dataset to identify the "red flags" that lead to a customer leaving and builds a machine learning model to predict which customers are at risk before they quit.

By identifying these customers early, the business can take proactive steps to keep them, saving money and improving long-term growth.

---

## Business Insights (The "So What?")
After analyzing 7,000+ customer records, several key patterns emerged. If you're a business manager, here is what the data is telling us:

1. **The "Contract" Trap**: Customers on **Month-to-Month contracts** are significantly more likely to leave than those on 1 or 2-year plans.
2. **The Fiber Optic Paradox**: Surprisingly, customers with **Fiber Optic internet** churn at a higher rate. This suggests there might be issues with service reliability or price-to-value perception in that specific segment.
3. **Payment Friction**: People using **Electronic Checks** leave more often than those on automatic bank transfers or credit cards.
4. **The "Newbie" Phase**: Most churn happens in the first few months. If a customer stays past the first year, they are much more likely to remain loyal for a long time.

---

## The Technical Solution
I built and compared three different AI models to see which could best predict churn:
* **Logistic Regression** (The Baseline)
* **Random Forest** (The Ensemble Approach)
* **XGBoost** (The Winner)

**Results:** The XGBoost model achieved an **AUC-ROC of 0.84**, meaning it is highly effective at distinguishing between loyal customers and those about to leave.

---

## Recommendations
To reduce the churn rate (currently 26.54%), the business should:
* **Incentivize Long-term Contracts**: Offer a small discount for switching from month-to-month to a 1-year plan.
* **Audit the Fiber Optic Experience**: Investigate why high-speed users are unhappy.
* **Promote Auto-Pay**: Give a one-time credit to customers who switch from electronic checks to automated billing.

---

## Project Structure
* `Raw_data.csv`: The original, untouched dataset.
* `cleaned_data.csv`: The data after handling missing values and formatting.
* `churn_analysis.py`: The Python "engine" that cleans data and trains the models.
* `models/`: Contains the saved "brains" (Pickle files) of the trained models.

---

## How to Run
1. Clone this repo.
2. Install requirements: `pip install pandas xgboost scikit-learn seaborn`
3. Run the analysis: `python churn_analysis.py`
