from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path("models/health_risk_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

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
        payload = request.get_json(force=True)

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

        row = np.array([values], dtype=float)
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
                "This tool is for screening support only and not a medical diagnosis.",
                "Use real patient or survey data to retrain the model before production deployment.",
                "Consider adding user authentication, database logging, and audit trails for real deployments."
            ]
        }
        return jsonify(response)

    except ValueError:
        return jsonify({"error": "All inputs must be numeric."}), 400
    except Exception as exc:
        return jsonify({"error": f"Unexpected server error: {str(exc)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
