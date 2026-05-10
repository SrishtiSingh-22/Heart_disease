"""
Heart Disease Prediction Model Trainer
Dataset: UCI Heart Disease Dataset (loaded from sklearn / built-in sample)
Run this script once to generate heart_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
# UCI Heart Disease Dataset columns:
# age, sex, cp, trestbps, chol, fbs, restecg,
# thalach, exang, oldpeak, slope, ca, thal, target

url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_data.csv"

try:
    df = pd.read_csv(url)
    print(f"✅ Dataset loaded from URL: {df.shape}")
except Exception:
    # Fallback: generate realistic synthetic data based on UCI stats
    print("⚠️  URL failed — generating synthetic UCI-like dataset...")
    np.random.seed(42)
    n = 303
    df = pd.DataFrame({
        'age':      np.random.normal(54, 9, n).clip(29, 77).astype(int),
        'sex':      np.random.binomial(1, 0.68, n),
        'cp':       np.random.choice([0,1,2,3], n, p=[0.47,0.17,0.28,0.08]),
        'trestbps': np.random.normal(131, 17, n).clip(94, 200).astype(int),
        'chol':     np.random.normal(246, 52, n).clip(126, 564).astype(int),
        'fbs':      np.random.binomial(1, 0.15, n),
        'restecg':  np.random.choice([0,1,2], n, p=[0.48,0.49,0.03]),
        'thalach':  np.random.normal(150, 23, n).clip(71, 202).astype(int),
        'exang':    np.random.binomial(1, 0.33, n),
        'oldpeak':  np.abs(np.random.normal(1.0, 1.2, n)).clip(0, 6.2).round(1),
        'slope':    np.random.choice([0,1,2], n, p=[0.21,0.46,0.33]),
        'ca':       np.random.choice([0,1,2,3], n, p=[0.58,0.22,0.13,0.07]),
        'thal':     np.random.choice([1,2,3], n, p=[0.06,0.55,0.39]),
    })
    # Create target correlated with risk factors
    risk_score = (
        (df['age'] > 55).astype(int) * 1.5 +
        df['sex'] * 0.5 +
        (df['cp'] == 0).astype(int) * 2.0 +
        (df['trestbps'] > 140).astype(int) * 1.0 +
        (df['chol'] > 240).astype(int) * 0.8 +
        (df['thalach'] < 140).astype(int) * 1.2 +
        df['exang'] * 1.5 +
        (df['oldpeak'] > 1.5).astype(int) * 1.3 +
        (df['ca'] > 0).astype(int) * 1.8
    )
    df['target'] = (risk_score + np.random.normal(0, 1, n) > 5).astype(int)
    print(f"✅ Synthetic dataset created: {df.shape}")

# ─────────────────────────────────────────────
# 2. Data Cleaning
# ─────────────────────────────────────────────
print(f"\n📊 Missing values:\n{df.isnull().sum()}")
df.dropna(inplace=True)

# Ensure target is binary (0 or 1)
if 'target' in df.columns:
    df['target'] = (df['target'] > 0).astype(int)
elif 'condition' in df.columns:
    df.rename(columns={'condition': 'target'}, inplace=True)
    df['target'] = (df['target'] > 0).astype(int)

print(f"\n✅ Clean dataset: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# ─────────────────────────────────────────────
# 3. Features & Split
# ─────────────────────────────────────────────
FEATURES = ['age','sex','cp','trestbps','chol','fbs',
            'restecg','thalach','exang','oldpeak','slope','ca','thal']

X = df[FEATURES]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 4. Scaling
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. Train & Compare Models
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "accuracy": acc}
    print(f"\n🔹 {name}")
    print(f"   Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["No Disease","Heart Disease"]))

# ─────────────────────────────────────────────
# 6. Select Best Model
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
best_acc = results[best_name]["accuracy"]

print(f"\n🏆 Best Model: {best_name} ({best_acc*100:.2f}% accuracy)")

# ─────────────────────────────────────────────
# 7. Save Model + Scaler + Metadata
# ─────────────────────────────────────────────
import os
os.makedirs(os.path.dirname(__file__) or '.', exist_ok=True)

joblib.dump(best_model, "model/heart_model.pkl")
joblib.dump(scaler,     "model/scaler.pkl")

metadata = {
    "model_name": best_name,
    "accuracy":   round(best_acc * 100, 2),
    "features":   FEATURES,
    "all_accuracies": {k: round(v["accuracy"]*100, 2) for k, v in results.items()}
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Saved: model/heart_model.pkl")
print("✅ Saved: model/scaler.pkl")
print("✅ Saved: model/metadata.json")
print("\n🚀 Ready! Now run: python app.py")
