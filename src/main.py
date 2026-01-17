import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------
# PAGE CONFIG
# ------------------------------------
st.set_page_config(
    page_title="Digital Wallet ML System",
    layout="wide"
)

# ------------------------------------
# LOAD DATA
# ------------------------------------
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "..", "data", "digital_wallet_transactions.csv")

    df = pd.read_csv(path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    user_df = df.groupby("user_id").agg(
        total_spend=("product_amount", "sum"),
        transaction_count=("transaction_id", "count"),
        avg_transaction_amount=("product_amount", "mean"),
        active_days=("transaction_date", "nunique")
    ).reset_index()

    return user_df

user_df = load_data()



# ------------------------------------
# FEATURE ENGINEERING
# ------------------------------------
threshold = user_df["total_spend"].quantile(0.7)
user_df["high_value_user"] = (user_df["total_spend"] >= threshold).astype(int)

features = [
    "transaction_count",
    "avg_transaction_amount",
    "active_days"
]

X = user_df[features]
y_reg = user_df["total_spend"]
y_clf = user_df["high_value_user"]

# ------------------------------------
# SCALING
# ------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_scaled, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# ------------------------------------
# MODELS (IDENTICAL LOGIC)
# ------------------------------------
lr = LinearRegression()
lr.fit(Xr_train, yr_train)
yr_pred = lr.predict(Xr_test)

logreg = LogisticRegression(max_iter=500)
logreg.fit(Xc_train, yc_train)
log_pred = logreg.predict(Xc_test)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(Xc_train, yc_train)
knn_pred = knn.predict(Xc_test)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(Xc_train, yc_train)
dt_pred = dt.predict(Xc_test)

# ------------------------------------
# RMSE Squared
# ------------------------------------

rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))

# ------------------------------------
# HEADER
# ------------------------------------
st.title("💳 Digital Wallet Machine Learning Dashboard")
st.caption("Built by Bubacarr Bah | Data Scientist & Machine Learning Engineer")

# ------------------------------------
# KPI METRICS
# ------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Users", len(user_df))
c2.metric("High-Value Users", user_df["high_value_user"].sum())
c3.metric("Average Spend", f"{user_df['total_spend'].mean():.2f}")
c4.metric("Avg Transactions", f"{user_df['transaction_count'].mean():.1f}")

st.divider()

# ------------------------------------
# User-Level Features DISPLAY
# ------------------------------------

st.subheader("👤 User-Level Features (After Aggregation)")

st.markdown("""
Each row below represents **one user** after aggregating raw transactions.
These features are used for **EDA, modeling, and predictions**.
""")

st.dataframe(
    user_df.head(10),
    use_container_width=True
)
# ------------------------------------
# EDA SECTION
# ------------------------------------
st.subheader("📈 Exploratory Data Analysis")

c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots()
    sns.histplot(user_df["total_spend"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Total Spend")
    st.pyplot(fig)

with c2:
    fig, ax = plt.subplots()
    sns.heatmap(
        user_df[features + ["total_spend"]].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

st.divider()

# ------------------------------------
# REGRESSION RESULTS
# ------------------------------------
st.subheader("💰 Total Spend Prediction (Regression)")

c1, c2 = st.columns(2)

with c1:
    st.write("**Model Performance**")
    st.write("MAE:", round(mean_absolute_error(yr_test, yr_pred), 2))
    st.write("RMSE:", round(rmse, 2))
    st.write("R² Score:", round(r2_score(yr_test, yr_pred), 3))

with c2:
    fig, ax = plt.subplots()
    sns.scatterplot(x=yr_test, y=yr_pred, ax=ax)
    ax.plot(
        [yr_test.min(), yr_test.max()],
        [yr_test.min(), yr_test.max()],
        "--", color="red"
    )
    ax.set_xlabel("Actual Spend")
    ax.set_ylabel("Predicted Spend")
    st.pyplot(fig)

st.divider()

# ------------------------------------
# CLASSIFICATION RESULTS
# ------------------------------------
st.subheader("🎯 High-Value User Classification")

tabs = st.tabs(["Logistic", "KNN", "Decision Tree"])

models = {
    "Logistic": log_pred,
    "KNN": knn_pred,
    "Decision Tree": dt_pred
}

for tab, (name, preds) in zip(tabs, models.items()):
    with tab:
        st.text(classification_report(yc_test, preds))
        fig, ax = plt.subplots()
        sns.heatmap(
            confusion_matrix(yc_test, preds),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )
        ax.set_title(f"{name} Confusion Matrix")
        st.pyplot(fig)

st.divider()

# ------------------------------------
# CLUSTERING
# ------------------------------------
st.subheader("🧩 User Segmentation (K-Means)")

kmeans = KMeans(n_clusters=3, random_state=42)
user_df["cluster"] = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots()
sns.scatterplot(
    data=user_df,
    x="transaction_count",
    y="total_spend",
    hue="cluster",
    palette="Set2",
    ax=ax
)
ax.set_title("User Segments by Behavior")
st.pyplot(fig)

st.divider()

# ------------------------------------
# NEW USER PREDICTION
# ------------------------------------
st.subheader("🧪 New User Prediction")

c1, c2, c3 = st.columns(3)
tx = c1.number_input("Transaction Count", min_value=1, value=3)
avg = c2.number_input("Average Transaction Amount", min_value=1.0, value=5000.00)
days = c3.number_input("Active Days", min_value=1, value=5)

new_user = scaler.transform([[tx, avg, days]])

if st.button("Predict User"):
    spend_pred = lr.predict(new_user)[0]
    value_pred = logreg.predict(new_user)[0]

    st.success(f"Predicted Total Spend: {spend_pred:.2f}")
    st.info("High-Value User" if value_pred == 1 else "Low-Value User")

# ------------------------------------
# BUSINESS INSIGHTS
# ------------------------------------
st.divider()
st.subheader("📌 Business Insights")

st.markdown("""
• Transaction frequency and average transaction amount are the **strongest drivers of total spend**  
• High-value users are typically **active over more days**, not just high spenders once  
• Clustering reveals **distinct user segments**, enabling targeted promotions  
• Predictive models allow **early identification of valuable users** for retention strategies  
• This pipeline supports **data-driven marketing, personalization, and churn prevention**
""")

st.markdown("---")
st.markdown(
    "© 2025 **Bubacarr Bah** · Digital Wallet ML System · "
   
)
