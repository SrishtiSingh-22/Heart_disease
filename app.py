"""
Heart Disease Prediction – Flask Backend
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load model, scaler, metadata on startup
# ─────────────────────────────────────────────
BASE = os.path.dirname(__file__)

model   = joblib.load(os.path.join(BASE, "model", "heart_model.pkl"))
scaler  = joblib.load(os.path.join(BASE, "model", "scaler.pkl"))

with open(os.path.join(BASE, "model", "metadata.json")) as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
print(f"✅ Model loaded: {metadata['model_name']} ({metadata['accuracy']}% accuracy)")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", metadata=metadata)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract & validate features
        values = []
        for feat in FEATURES:
            val = data.get(feat)
            if val is None:
                return jsonify({"error": f"Missing field: {feat}"}), 400
            values.append(float(val))

        # Scale + predict
        arr = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        prediction = int(model.predict(arr_scaled)[0])
        probability = float(model.predict_proba(arr_scaled)[0][1]) * 100

        # Risk level
        if probability < 30:
            risk_level = "Low Risk"
            risk_class = "low"
        elif probability < 60:
            risk_level = "Moderate Risk"
            risk_class = "moderate"
        else:
            risk_level = "High Risk"
            risk_class = "high"

        return jsonify({
            "prediction":  prediction,
            "probability": round(probability, 1),
            "risk_level":  risk_level,
            "risk_class":  risk_class,
            "model_used":  metadata["model_name"]
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info")
def model_info():
    return jsonify(metadata)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
