from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from schemas import TransactionInput, PredictionResponse, ThresholdAnalysisRequest, ThresholdAnalysisResponse
from features import engineer_features
from model import get_model

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring with SHAP explainability and business cost analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model performance metrics from 05_evaluation.ipynb
# Update these after running the final evaluation notebook
MODEL_METRICS = {
    "precision": 0.0,   # placeholder — fill after training
    "recall": 0.0,
    "fpr": 0.0,         # false positive rate at operating threshold
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    try:
        model = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    feature_vector = engineer_features(transaction.model_dump())
    result = model.predict(feature_vector)

    daily_volume = 1_000_000
    fpr = MODEL_METRICS["fpr"] if MODEL_METRICS["fpr"] > 0 else 0.01
    business_impact = _compute_business_impact(
        daily_volume=daily_volume,
        fpr=fpr,
        cost_per_fp=10.0,
        cost_per_fn=150.0,
    )

    return PredictionResponse(**result, business_impact=business_impact)


@app.post("/threshold-analysis", response_model=ThresholdAnalysisResponse)
def threshold_analysis(req: ThresholdAnalysisRequest):
    """
    Lets the frontend calculate the real-world business cost of a threshold choice.
    Uses model performance metrics from evaluation to project FP/FN counts.
    """
    fpr = MODEL_METRICS["fpr"] if MODEL_METRICS["fpr"] > 0 else 0.01
    fnr = 1 - (MODEL_METRICS["recall"] if MODEL_METRICS["recall"] > 0 else 0.80)
    fraud_rate = 0.035  # IEEE-CIS dataset fraud rate

    legitimate_per_day = int(req.daily_transaction_volume * (1 - fraud_rate))
    fraud_per_day = int(req.daily_transaction_volume * fraud_rate)

    # Threshold adjustment: higher threshold → fewer FPs but more FNs
    threshold_factor = req.threshold / 0.5
    adjusted_fpr = min(fpr * (1 / threshold_factor), 1.0)
    adjusted_fnr = min(fnr * threshold_factor, 1.0)

    fp_per_day = int(legitimate_per_day * adjusted_fpr)
    fn_per_day = int(fraud_per_day * adjusted_fnr)

    tp_per_day = fraud_per_day - fn_per_day
    fp_fp_per_day_total = fp_per_day + tp_per_day
    precision = tp_per_day / max(fp_fp_per_day_total, 1)
    recall = tp_per_day / max(fraud_per_day, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    fp_cost = fp_per_day * req.cost_per_false_positive
    fn_cost = fn_per_day * req.cost_per_false_negative

    return ThresholdAnalysisResponse(
        threshold=req.threshold,
        estimated_false_positives_per_day=fp_per_day,
        estimated_false_negatives_per_day=fn_per_day,
        estimated_daily_fp_cost=round(fp_cost, 2),
        estimated_daily_fn_cost=round(fn_cost, 2),
        estimated_total_daily_cost=round(fp_cost + fn_cost, 2),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


@app.get("/model-info")
def model_info():
    return {
        "version": "1.0.0",
        "architecture": "LightGBM + MLP stacked ensemble with Logistic Regression meta-learner",
        "dataset": "IEEE-CIS Fraud Detection (Kaggle)",
        "training_approach": "Time-based split — no data leakage",
        "operating_threshold": 0.70,
        "review_threshold": 0.35,
        "metrics": MODEL_METRICS,
        "explainability": "SHAP TreeExplainer",
    }


def _compute_business_impact(daily_volume, fpr, cost_per_fp, cost_per_fn):
    fraud_rate = 0.035
    legitimate = int(daily_volume * (1 - fraud_rate))
    fp_per_day = int(legitimate * fpr)
    return {
        "daily_transaction_volume": daily_volume,
        "false_positives_per_day": fp_per_day,
        "daily_fp_cost_usd": round(fp_per_day * cost_per_fp, 2),
        "customers_affected_per_day": fp_per_day,
    }
