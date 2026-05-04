import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "health_risk_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "blood_sugar",
    "cholesterol",
    "activity_level",
    "smoking",
    "family_history",
]


def risk_level(probability: float) -> str:
    if probability < 0.35:
        return "Low"
    if probability < 0.70:
        return "Moderate"
    return "High"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        values = []
        missing = []
        for feature in FEATURES:
            if feature not in payload:
                missing.append(feature)
            else:
                values.append(float(payload[feature]))

        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        age, bmi, sys_bp, dia_bp, sugar, chol, activity, smoking, family_history = values

        rules = [
            (18 <= age <= 100, "Age must be between 18 and 100."),
            (10 <= bmi <= 60, "BMI must be between 10 and 60."),
            (70 <= sys_bp <= 250, "Systolic BP must be between 70 and 250."),
            (40 <= dia_bp <= 150, "Diastolic BP must be between 40 and 150."),
            (50 <= sugar <= 400, "Blood sugar must be between 50 and 400."),
            (80 <= chol <= 400, "Cholesterol must be between 80 and 400."),
            (0 <= activity <= 7, "Activity level must be between 0 and 7."),
            (smoking in [0, 1], "Smoking must be 0 or 1."),
            (family_history in [0, 1], "Family history must be 0 or 1.")
        ]

        for valid, message in rules:
            if not valid:
                return jsonify({"error": message}), 400

        row = pd.DataFrame([values], columns=FEATURES, dtype=float)
        row_scaled = scaler.transform(row)
        probability = float(model.predict_proba(row_scaled)[0][1])
        prediction = int(probability >= 0.5)

        response = {
            "prediction": prediction,
            "risk_level": risk_level(probability),
            "risk_score": round(probability * 100, 2),
            "summary": (
                "Higher likelihood of health risk. Follow up with a qualified health professional."
                if prediction == 1
                else "Lower likelihood of immediate health risk based on the entered values."
            ),
            "notes": [
                "This demo tool is not a medical diagnosis or clinical decision system.",
                "The included model is trained on synthetic data and is not clinically validated.",
                "Retrain and validate with appropriate real-world data before any production health use."
            ]
        }
        return jsonify(response)

    except ValueError:
        return jsonify({"error": "All inputs must be numeric."}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Unexpected server error."}), 500


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(debug=debug)
