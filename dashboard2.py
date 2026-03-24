import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("fraud_dataset.csv")

# Small sample for clarity
df = df.sample(5000)

# -----------------------------
# TITLE
# -----------------------------
st.title("💳 UPI Fraud Detection Dashboard (Simple View)")

# -----------------------------
# KPI SECTION
# -----------------------------
total = len(df)
fraud = df["isFraud"].sum()
amount = df["amount"].sum()

st.write(f"Total Transactions: {total}")
st.write(f"Fraud Transactions: {fraud}")
st.write(f"Total Amount: ₹{int(amount)}")

# -----------------------------
# GRAPH 1: FRAUD COUNT
# -----------------------------
st.subheader("1. Fraud vs Normal")

fig1, ax1 = plt.subplots()
sns.countplot(x="isFraud", data=df)
st.pyplot(fig1)

st.write("👉 Most transactions are normal, fraud cases are fewer.")

# -----------------------------
# GRAPH 2: AMOUNT DISTRIBUTION
# -----------------------------
st.subheader("2. Transaction Amount")

fig2, ax2 = plt.subplots()
df["amount"].plot(kind="hist", bins=30)
st.pyplot(fig2)

st.write("👉 Most transactions are small amounts.")

# -----------------------------
# GRAPH 3: TRANSACTION TYPE
# -----------------------------
st.subheader("3. Transaction Type vs Fraud")

fig3, ax3 = plt.subplots()
sns.countplot(x="type", hue="isFraud", data=df)
plt.xticks(rotation=45)
st.pyplot(fig3)

st.write("👉 Some transaction types have higher fraud chances.")

# -----------------------------
# GRAPH 4: HEATMAP
# -----------------------------
st.subheader("4. Data Relationship")

fig4, ax4 = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True)
st.pyplot(fig4)

st.write("👉 Strong relationships help detect fraud patterns.")

st.success("✅ Simple Dashboard Ready")