import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

n = 1200
age = np.random.randint(18, 81, n)
bmi = np.round(np.random.normal(27, 5, n).clip(15, 45), 1)
systolic_bp = np.round(np.random.normal(128, 18, n).clip(85, 220), 0)
diastolic_bp = np.round(np.random.normal(82, 12, n).clip(50, 140), 0)
blood_sugar = np.round(np.random.normal(108, 35, n).clip(60, 320), 0)
cholesterol = np.round(np.random.normal(195, 42, n).clip(100, 360), 0)
activity_level = np.random.randint(0, 8, n)
smoking = np.random.binomial(1, 0.18, n)
family_history = np.random.binomial(1, 0.35, n)

score = (
    0.025 * (age - 40)
    + 0.09 * (bmi - 25)
    + 0.03 * (systolic_bp - 120)
    + 0.025 * (diastolic_bp - 80)
    + 0.035 * (blood_sugar - 100)
    + 0.018 * (cholesterol - 180)
    - 0.35 * activity_level
    + 1.0 * smoking
    + 0.8 * family_history
    + np.random.normal(0, 1.3, n)
)

risk = (score > 3.8).astype(int)

df = pd.DataFrame({
    "age": age,
    "bmi": bmi,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "blood_sugar": blood_sugar,
    "cholesterol": cholesterol,
    "activity_level": activity_level,
    "smoking": smoking,
    "family_history": family_history,
    "risk": risk
})

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

df.to_csv(DATA_DIR / "health_risk_dataset.csv", index=False)

X = df.drop(columns=["risk"])
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

joblib.dump(model, MODELS_DIR / "health_risk_model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

print("Synthetic demo training complete.")
print("Files saved successfully.")
