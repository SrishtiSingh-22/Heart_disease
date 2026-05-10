# 💗 Heart Disease Prediction System

AI-powered cardiac risk assessment using scikit-learn + Flask.

---

## 🚀 Quick Start (3 steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model (sirf ek baar)
```bash
python model/train_model.py
```
Ye 3 files create karega:
- `model/heart_model.pkl`  → trained ML model
- `model/scaler.pkl`       → feature scaler
- `model/metadata.json`    → accuracy + info

### Step 3 — Flask server chalao
```bash
python app.py
```
Browser mein kholo: **http://localhost:5000**

---

## 📁 Project Structure

```
heart-disease-predictor/
├── model/
│   ├── train_model.py      ← ML training script
│   ├── heart_model.pkl     ← saved model (auto-generated)
│   ├── scaler.pkl          ← feature scaler (auto-generated)
│   └── metadata.json       ← model info (auto-generated)
├── templates/
│   └── index.html          ← frontend UI
├── app.py                  ← Flask backend
├── requirements.txt        ← Python dependencies
└── README.md
```

---

## 🧪 API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET  | `/`           | Web UI |
| POST | `/predict`    | JSON prediction |
| GET  | `/model-info` | Model metadata |

### POST /predict — Example
```json
{
  "age": 54, "sex": 1, "cp": 0,
  "trestbps": 140, "chol": 250,
  "fbs": 0, "restecg": 1,
  "thalach": 130, "exang": 1,
  "oldpeak": 2.0, "slope": 1,
  "ca": 1, "thal": 2
}
```

### Response
```json
{
  "prediction": 1,
  "probability": 72.4,
  "risk_level": "High Risk",
  "risk_class": "high",
  "model_used": "Logistic Regression"
}
```

---

## 📊 Features (13 UCI parameters)

| Feature | Description | Range |
|---------|-------------|-------|
| age | Patient age | 29–77 |
| sex | 1=Male, 0=Female | 0/1 |
| cp | Chest pain type | 0–3 |
| trestbps | Resting blood pressure (mm Hg) | 94–200 |
| chol | Serum cholesterol (mg/dl) | 126–564 |
| fbs | Fasting blood sugar >120 mg/dl | 0/1 |
| restecg | Resting ECG results | 0–2 |
| thalach | Max heart rate achieved | 71–202 |
| exang | Exercise-induced angina | 0/1 |
| oldpeak | ST depression (exercise vs rest) | 0–6.2 |
| slope | Slope of peak exercise ST | 0–2 |
| ca | Major vessels (fluoroscopy) | 0–3 |
| thal | Thalassemia type | 1–3 |

---

## 🤖 ML Models Compared

- **Logistic Regression** — Best for beginners, ~80–85% accuracy on UCI
- **Decision Tree** — Easy to interpret
- **Random Forest** — Often most accurate, robust

Best model auto-selected and saved.

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**.  
It is NOT a medical device and should NOT replace professional medical diagnosis.

---

## 🔧 Optional Upgrades

- [ ] MySQL database for prediction history
- [ ] Login system (Flask-Login)
- [ ] Graphs with Chart.js
- [ ] Deploy on Render / Railway / Heroku
