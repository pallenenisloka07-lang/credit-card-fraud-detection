# Credit Card Fraud Detection - Datathon Version

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# -------------------------
# Load Dataset
# -------------------------
data = pd.read_csv("cleaned_creditcard.csv")


# -------------------------
# Feature Engineering
# -------------------------
# log transform for transaction amount
data["Amount_log"] = np.log1p(data["Amount"])

# extract hour from transaction time
data["Hour"] = (data["Time"] // 3600) % 24


# -------------------------
# Features and Target
# -------------------------
X = data.drop("Class", axis=1)
y = data["Class"]


# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------
# Handle Class Imbalance
# -------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


# -------------------------
# Feature Scaling
# -------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------------
# Model (XGBoost)
# -------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -------------------------
# Predictions
# -------------------------
y_prob = model.predict_proba(X_test)[:,1]

# threshold tuning (detect more fraud)
y_pred = (y_prob > 0.3).astype(int)


# -------------------------
# Evaluation
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", auc)


# -------------------------
# Fraud Probability Example
# -------------------------
print("\nSample Fraud Probabilities:")
print(y_prob[:10])


# -------------------------
# Feature Importance
# -------------------------
importances = model.feature_importances_

plt.figure(figsize=(10,5))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")

# save graph instead of showing it
plt.savefig("feature_importance.png")

print("\nFeature importance graph saved as feature_importance.png")