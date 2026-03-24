import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# SETUP
# -----------------------------
sns.set(style="whitegrid")
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("fraud_dataset.csv")   # 👈 dataset ka naam change kar lena

print("Columns:", df.columns)

# -----------------------------
# BASIC CLEANING
# -----------------------------
df = df.dropna()

# -----------------------------
# 1. FRAUD vs NON-FRAUD
# -----------------------------
plt.figure()
sns.countplot(x="isFraud", data=df)
plt.title("Fraud vs Non-Fraud")
plt.savefig("outputs/fraud_count.png")
plt.close()

# -----------------------------
# 2. AMOUNT DISTRIBUTION
# -----------------------------
plt.figure()
df["amount"].plot(kind="hist", bins=30)
plt.title("Transaction Amount Distribution")
plt.savefig("outputs/amount_distribution.png")
plt.close()

# -----------------------------
# 3. TIME vs FRAUD
# -----------------------------
plt.figure()
df.groupby("step")["isFraud"].sum().plot()
plt.title("Fraud vs Time")
plt.savefig("outputs/time_vs_fraud.png")
plt.close()

# -----------------------------
# 4. TRANSACTION TYPE
# -----------------------------
plt.figure()
sns.countplot(x="type", hue="isFraud", data=df)
plt.title("Transaction Type vs Fraud")
plt.xticks(rotation=45)
plt.savefig("outputs/type_vs_fraud.png")
plt.close()

# -----------------------------
# 5. BALANCE SCATTER
# -----------------------------
plt.figure()
sns.scatterplot(x="oldbalanceOrg", y="newbalanceOrig", hue="isFraud", data=df)
plt.title("Balance Comparison")
plt.savefig("outputs/balance_scatter.png")
plt.close()

# -----------------------------
# 6. HEATMAP
# -----------------------------
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")
plt.close()

# -----------------------------
# 7. BOXPLOT
# -----------------------------
plt.figure()
sns.boxplot(x="isFraud", y="amount", data=df)
plt.title("Amount vs Fraud")
plt.savefig("outputs/boxplot.png")
plt.close()

# -----------------------------
# ML MODEL (Logistic Regression)
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Select features
X = df[["amount", "oldbalanceOrg", "newbalanceOrig"]]
y = df["isFraud"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("🔥 ALL GRAPHS + MODEL DONE SUCCESSFULLY")