from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import json

import joblib
import numpy as np
from typing import Optional

app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts presence of heart disease from 13 clinical features.",
    version="1.0.0",
)

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
templates = Jinja2Templates(directory="templates")


class PatientInput(BaseModel):
    age: int = Field(..., ge=20, le=80, description="Age in years", example=52)
    sex: int = Field(..., ge=0, le=1, description="1=male, 0=female", example=1)
    cp: int = Field(..., ge=0, le=3, description="Chest pain type 0-3", example=0)
    trestbps: int = Field(
        ..., ge=80, le=200, description="Resting BP (mmHg)", example=125
    )
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol", example=212)
    fbs: int = Field(..., ge=0, le=1, description="Fasting BS > 120? 1/0", example=0)
    restecg: int = Field(
        ..., ge=0, le=2, description="Resting ECG result 0-2", example=1
    )
    thalach: int = Field(
        ..., ge=60, le=220, description="Max heart rate achieved", example=168
    )
    exang: int = Field(
        ..., ge=0, le=1, description="Exercise-induced angina", example=0
    )
    oldpeak: float = Field(
        ..., ge=0.0, le=7.0, description="ST depression", example=1.0
    )
    slope: int = Field(..., ge=0, le=2, description="Slope of ST segment", example=2)
    ca: int = Field(..., ge=0, le=4, description="# major vessels (0-4)", example=2)
    thal: int = Field(
        ..., ge=0, le=3, description="Thal: 0=norm,1=fixed,2=rev", example=3
    )


class PredictionOut(BaseModel):
    prediction: str
    risk_score: float
    risk_category: str
    probabilities: dict
    model_used: str


def risk_category(prob: float) -> str:
    if prob < 0.30:
        return "low"
    if prob < 0.60:
        return "moderate"
    return "high"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def get_metrics():
    with open("reports/metrics.json") as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionOut)
def predict(patient: PatientInput):
    features = np.array(
        [
            [
                patient.age,
                patient.sex,
                patient.cp,
                patient.trestbps,
                patient.chol,
                patient.fbs,
                patient.restecg,
                patient.thalach,
                patient.exang,
                patient.oldpeak,
                patient.slope,
                patient.ca,
                patient.thal,
            ]
        ]
    )

    scaled = scaler.transform(features)
    pred = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]
    disease_prob = float(proba[1])

    with open("reports/metrics.json") as f:
        model_name = json.load(f)["model"]

    return PredictionOut(
        prediction="disease" if pred == 1 else "no disease",
        risk_score=round(disease_prob, 4),
        risk_category=risk_category(disease_prob),
        probabilities={
            "no_disease": round(float(proba[0]), 4),
            "disease": round(float(proba[1]), 4),
        },
        model_used=model_name,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
