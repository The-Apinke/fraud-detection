"""
Model loading, inference, and SHAP explanation.
Handles the stacked ensemble: LightGBM + MLP -> Logistic Regression meta-learner.
"""
import os
import pickle
import numpy as np
import shap
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

# Thresholds for verdict bucketing
BLOCK_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.35

# Expected model performance metrics loaded from evaluation
# These are populated after notebook 05_evaluation runs
EVAL_METRICS = {
    "precision_at_block": 0.0,
    "recall_at_block": 0.0,
    "f1_at_block": 0.0,
}


class FraudDetectionModel:
    def __init__(self):
        self.lgbm = None
        self.mlp = None
        self.meta_learner = None
        self.shap_explainer = None
        self.feature_names = None
        self._loaded = False

    def load(self):
        lgbm_path = MODEL_DIR / "lgbm_model.pkl"
        meta_path = MODEL_DIR / "meta_learner.pkl"
        feature_path = MODEL_DIR / "feature_names.pkl"

        if not lgbm_path.exists():
            raise FileNotFoundError(
                f"Model not found at {lgbm_path}. "
                "Run notebooks 03 and 04 first and download the artifacts."
            )

        with open(lgbm_path, "rb") as f:
            self.lgbm = pickle.load(f)

        with open(meta_path, "rb") as f:
            self.meta_learner = pickle.load(f)

        if feature_path.exists():
            with open(feature_path, "rb") as f:
                self.feature_names = pickle.load(f)

        # MLP is optional — ensemble degrades gracefully without it
        mlp_path = MODEL_DIR / "mlp_model.pkl"
        if mlp_path.exists():
            with open(mlp_path, "rb") as f:
                self.mlp = pickle.load(f)

        self.shap_explainer = shap.TreeExplainer(self.lgbm)
        self._loaded = True

    def predict(self, feature_vector: np.ndarray) -> dict:
        if not self._loaded:
            self.load()

        x = feature_vector.reshape(1, -1)

        lgbm_prob = self.lgbm.predict_proba(x)[0, 1]

        if self.mlp is not None:
            mlp_prob = self.mlp.predict_proba(x)[0, 1]
            meta_features = np.array([[lgbm_prob, mlp_prob]])
        else:
            meta_features = np.array([[lgbm_prob, lgbm_prob]])

        final_prob = float(self.meta_learner.predict_proba(meta_features)[0, 1])

        shap_values = self.shap_explainer.shap_values(x)
        # For binary classification, shap_values is list[pos_class, neg_class] or array
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        names = self.feature_names or [f"feature_{i}" for i in range(len(sv))]
        top_shap = sorted(
            [{"feature": n, "shap_value": float(v)} for n, v in zip(names, sv)],
            key=lambda d: abs(d["shap_value"]),
            reverse=True,
        )[:5]

        verdict = _get_verdict(final_prob)
        risk_score = int(round(final_prob * 100))

        return {
            "fraud_probability": final_prob,
            "risk_score": risk_score,
            "verdict": verdict,
            "threshold_used": BLOCK_THRESHOLD if verdict == "BLOCK" else REVIEW_THRESHOLD,
            "shap_explanations": top_shap,
        }


def _get_verdict(prob: float) -> str:
    if prob >= BLOCK_THRESHOLD:
        return "BLOCK"
    if prob >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "APPROVE"


# Singleton — loaded once at startup
_model_instance: FraudDetectionModel | None = None


def get_model() -> FraudDetectionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = FraudDetectionModel()
        _model_instance.load()
    return _model_instance
