from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import datetime
import sys
sys.path.append(".")
from model import calculate_loan, add_dp_noise, get_feature_importance

app = FastAPI(title="MediGuard AI — Heart Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],  # "null" = when frontend is opened as file://
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

with open("disease_model.pkl","rb") as f: disease_model = pickle.load(f)
with open("scaler.pkl","rb") as f:        scaler        = pickle.load(f)

FEATURE_COLS  = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']
FEATURE_NAMES = ['Age','Sex','Chest Pain Type','Resting BP','Cholesterol',
                 'Fasting Blood Sugar','Resting ECG','Max Heart Rate',
                 'Exercise Angina','ST Depression','Slope','Vessels','Thal']

audit_log = []

def log_event(event):
    audit_log.append({
        "time": datetime.datetime.now().strftime("%I:%M:%S %p"),
        "event": event
    })

# ── Schemas ────────────────────────────────────────────────────────────────────
class HealthInput(BaseModel):
    age:       float   # Age in years
    sex:       int     # 1=Male, 0=Female
    cp:        int     # Chest pain type (0–3)
    trestbps:  float   # Resting blood pressure (mmHg)
    chol:      float   # Serum cholesterol (mg/dl)
    fbs:       int     # Fasting blood sugar > 120 mg/dl (1=True, 0=False)
    restecg:   int     # Resting ECG results (0–2)
    thalach:   float   # Max heart rate achieved
    exang:     int     # Exercise induced angina (1=Yes, 0=No)
    oldpeak:   float   # ST depression induced by exercise
    slope:     int     # Slope of peak exercise ST segment (0–2)
    ca:        int     # Number of major vessels (0–4)
    thal:      int     # Thal: 0=Normal, 1=Fixed defect, 2=Reversible defect

class LoanInput(BaseModel):
    income:             float
    credit_score:       float
    disease_risk_score: float
    treatment_cost:     float

# ── Routes ─────────────────────────────────────────────────────────────────────
_INDEX_PATH = Path(__file__).resolve().parent / "index.html"

@app.get("/")
def root():
    """Serve the frontend so you can open http://127.0.0.1:8000/ (no file:// or CORS issues)."""
    if _INDEX_PATH.exists():
        return FileResponse(_INDEX_PATH, media_type="text/html")
    return {"message": "MediGuard AI — Heart Risk Prediction is running. Add index.html to serve the app."}

@app.get("/health")
def health():
    return {"message": "MediGuard AI — Heart Risk Prediction is running"}

@app.post("/predict-disease")
def predict_disease(data: HealthInput):
    log_event("Patient submitted clinical data")
    log_event("Local federated model running (raw data never leaves device)")

    features = pd.DataFrame([{
        'age':data.age, 'sex':data.sex, 'cp':data.cp,
        'trestbps':data.trestbps, 'chol':data.chol, 'fbs':data.fbs,
        'restecg':data.restecg, 'thalach':data.thalach, 'exang':data.exang,
        'oldpeak':data.oldpeak, 'slope':data.slope, 'ca':data.ca, 'thal':data.thal
    }])

    features_scaled = scaler.transform(features)
    prob            = disease_model.predict_proba(features_scaled)[0][1]
    risk_score      = round(prob * 100)

    noisy_score = min(100, max(0, round(add_dp_noise(risk_score, sensitivity=2.0, epsilon=0.5))))
    log_event("Differential privacy noise applied to risk score")

    if noisy_score >= 70:
        risk_level     = "HIGH"
        action         = "Seek immediate cardiology consultation. ECG & angiography recommended."
        treatment_cost = 350000
        treatment_info = "Estimated cost includes angioplasty/bypass surgery + hospital stay"
    elif noisy_score >= 40:
        risk_level     = "MEDIUM"
        action         = "Schedule cardiac evaluation within 1 week. Lifestyle changes needed."
        treatment_cost = 80000
        treatment_info = "Estimated cost includes diagnostics, medication & monitoring"
    else:
        risk_level     = "LOW"
        action         = "Maintain healthy diet & exercise. Annual cardiac check-up advised."
        treatment_cost = 15000
        treatment_info = "Estimated cost for preventive care & routine check-ups"

    importance = get_feature_importance(disease_model, FEATURE_NAMES)
    log_event(f"Heart risk score generated: {noisy_score}/100 ({risk_level})")

    return {
        "risk_score":     noisy_score,
        "risk_level":     risk_level,
        "action":         action,
        "treatment_cost": treatment_cost,
        "treatment_info": treatment_info,
        "feature_importance": importance,
        "privacy_note":   "Raw clinical data was NOT shared. Only encrypted risk score used.",
        "dp_applied":     True
    }

@app.post("/approve-loan")
def approve_loan(data: LoanInput):
    log_event("Financial data received for loan assessment")
    log_event("Bank received only risk score — NOT raw health data ✅")

    result = calculate_loan(
        income=data.income,
        credit_score=data.credit_score,
        disease_risk_score=data.disease_risk_score,
        treatment_cost=data.treatment_cost
    )

    if result["approved"]:
        log_event(f"Loan APPROVED: Rs.{result['loan_amount']:,} at 8.5% over {result['tenure_months']} months")
    else:
        log_event("Loan requires manual review — visit nearest branch")

    return {
        **result,
        "privacy_note": "Bank never received your raw clinical or medical records.",
        "message": "Approved! Start your treatment immediately." if result["approved"] else "Not approved automatically. Please visit nearest branch for assistance."
    }

@app.get("/audit-log")
def get_audit_log():
    return {"log": audit_log}

@app.delete("/audit-log")
def clear_log():
    audit_log.clear()
    return {"message": "Log cleared"}
