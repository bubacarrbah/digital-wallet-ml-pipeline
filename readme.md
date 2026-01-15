# 📊 Digital Wallet User Analytics & Prediction

**Author:** Bubacarr Bah  
**Role:** Data Scientist / Machine Learning Engineer  


## Project Overview
This project analyzes digital wallet user behavior to predict **total spend** and classify **high-value users** using machine learning. The goal is to help businesses **identify valuable users**, optimize marketing campaigns and make data-driven decisions.

The project includes:  
- **Regression:** Predict total spend per user.  
- **Classification:** Identify high-value users using Logistic Regression, KNN, and Decision Trees.  
- **User-level insights:** Aggregated user metrics such as total spend, transaction count, average transaction amount, and active days.  
- **Interactive dashboard:** Built with Streamlit for visual exploration and new user predictions.  

---

## Data Features
| Feature | Description |
|---------|-------------|
| `user_id` | Unique identifier for each user |
| `total_spend` | Total amount spent by the user |
| `transaction_count` | Number of transactions made |
| `avg_transaction_amount` | Average amount spent per transaction |
| `active_days` | Number of unique days the user was active |
| `high_value_user` | 1 if user is in the top 30% spenders, else 0 |

These **user-level features** are aggregated from transaction-level data for **per-user analysis** and predictions.

---

## Exploratory Data Analysis (EDA)
EDA helps understand feature **distributions and relationships**:  

- **Histograms**: Show distribution of total spend, average transaction amount, transaction count, and active days.  
- **Correlation heatmap**: Highlights relationships between features. For example, total spend correlates strongly with average transaction amount and active days.  
- **Scatter plots**: Reveal patterns for high-value users visually.

**Why important:**  
- Ensures features are suitable for modeling.  
- Helps in feature selection and business interpretation.


---

## Regression: Predicting Total Spend
- **Model:** Linear Regression  
- **Performance Metrics:**  
  - MAE: ~634  
  - RMSE: ~X (insert your results)  
  - R²: ~X  

The regression model predicts **total spend** per user. A low MAE indicates **high accuracy**.  

**Example prediction:**  
- User with avg transaction = 3833, transaction count = 4, active days = 4  
- **Predicted total spend:** 15333.8 (matches observed spend)



---

## Classification: High-Value User
- **Target:** Top 30% spenders classified as high-value users.  
- **Models:**  
  1. **Logistic Regression** – Interpretable, good for small datasets, provides probabilities.  
  2. **K-Nearest Neighbors (KNN)** – Non-linear, flexible, slightly higher accuracy.  
  3. **Decision Tree** – Provides actionable rules and feature importance.

### Confusion Matrices
| Model | Confusion Matrix |
|-------|-----------------|
| Logistic Regression | 548  3<br>2  234 |
| KNN | 550 1<br>1 235 |
| Decision Tree | 548 3<br>1 235 |

**Interpretation:**  
- **TP:** High-value users correctly predicted  
- **TN:** Low-value users correctly predicted  
- **FP:** Low-value users predicted as high  
- **FN:** High-value users predicted as low  

**Why Logistic Regression:**  
- Small dataset: avoids overfitting  
- Probabilistic output: easier for threshold-based decisions  
- Interpretability: clear understanding of feature impact  

**Decision Tree Feature Importance:**  
| Feature | Importance |
|---------|------------|
| Avg Transaction Amount | 0.6 |
| Active Days | 0.3 |
| Transaction Count | 0.1 |

---

## K-Means Clustering
- **Goal:** Segment users into 3 groups based on spending behavior.  
- **Insight:** Identifies clusters like low-value, medium-value, and high-value users.  


---

## New User Prediction
The Streamlit dashboard allows input of:  
- Average Transaction Amount  
- Transaction Count  
- Active Days  

**Outputs:**  
1. **Total Spend (Regression)** – Numeric prediction  
2. **High-Value Classification (Logistic Regression)** – Yes/No  

**Example:**  
- Input: avg transaction = 3833, transaction count = 4, active days = 4  
- Output: Predicted spend = 15333.8, High-value = Yes  

**How it works:**  
- Regression predicts spend using linear relationships.  
- Logistic Regression calculates the probability of being high-value; threshold = 0.7 quantile.

---

## Business Insights
- Users with **higher avg transaction amounts** and **more active days** are likely high-value.  
- **Repeat active users**, even with moderate spend, are high-value.  
- Decision Tree rules help define **thresholds for campaigns**.  
- Clustering helps segment users for **personalized strategies**.  

---

---

## 📜 License & Ownership

© 2025 Bubacarr Bah.  
This project is provided for educational and portfolio purposes.

You may reference this project, but redistribution or commercial use
requires explicit permission from the author.

