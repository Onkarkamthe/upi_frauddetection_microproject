import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Fraud Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("fraud_dataset.csv")

# -----------------------------
# BASIC CLEAN
# -----------------------------
df = df.sample(10000)  # 👈 IMPORTANT (data kam karke fast + clean)

# -----------------------------
# TITLE
# -----------------------------
st.title("💳 UPI Fraud Detection Dashboard")

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", len(df))
col2.metric("Fraud Cases", int(df["isFraud"].sum()))
col3.metric("Total Amount", int(df["amount"].sum()))

# -----------------------------
# GRAPH 1: FRAUD COUNT
# -----------------------------
st.subheader("Fraud vs Normal Transactions")

fig1, ax1 = plt.subplots()
sns.countplot(x="isFraud", data=df)
st.pyplot(fig1)

# -----------------------------
# GRAPH 2: AMOUNT DISTRIBUTION
# -----------------------------
st.subheader("Transaction Amount Distribution")

fig2, ax2 = plt.subplots()
df["amount"].plot(kind="hist", bins=30)
st.pyplot(fig2)

# -----------------------------
# GRAPH 3: TRANSACTION TYPE
# -----------------------------
st.subheader("Transaction Type vs Fraud")

fig3, ax3 = plt.subplots()
sns.countplot(x="type", hue="isFraud", data=df)
plt.xticks(rotation=45)
st.pyplot(fig3)

# -----------------------------
# GRAPH 4: HEATMAP
# -----------------------------
st.subheader("Correlation Heatmap")

fig4, ax4 = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True)
st.pyplot(fig4)

st.success("✅ Simple Dashboard Loaded Successfully")