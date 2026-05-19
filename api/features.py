"""
Feature engineering logic that mirrors the notebook pipeline.
This must stay in sync with 02_feature_engineering.ipynb.
"""
import numpy as np
import pandas as pd
from typing import Optional

HIGH_RISK_EMAIL_DOMAINS = {"anonymous.com", "aim.com", "protonmail.com"}
HIGH_RISK_PRODUCT_CODES = {"H", "C"}

KNOWN_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "icloud.com", "aol.com", "comcast.net", "msn.com",
}


def engineer_features(raw: dict) -> np.ndarray:
    """
    Transform raw transaction input into the feature vector the model expects.
    Returns a 1D numpy array ordered to match training columns.
    """
    features = {}

    features["transaction_amt"] = raw["transaction_amt"]
    features["transaction_amt_log"] = np.log1p(raw["transaction_amt"])

    features["amt_vs_mean_ratio"] = raw.get("amt_vs_mean_ratio", 1.0)
    features["amt_deviation"] = np.log1p(abs(raw["transaction_amt"] * raw.get("amt_vs_mean_ratio", 1.0) - raw["transaction_amt"]))

    features["txn_count_1h"] = raw.get("txn_count_1h", 0)
    features["txn_count_24h"] = raw.get("txn_count_24h", 0)
    features["txn_velocity_ratio"] = (
        features["txn_count_1h"] / max(features["txn_count_24h"], 1)
    )

    features["hour_of_day"] = raw["hour_of_day"]
    features["day_of_week"] = raw["day_of_week"]
    features["is_night"] = int(raw["hour_of_day"] < 6 or raw["hour_of_day"] >= 23)
    features["is_weekend"] = int(raw["day_of_week"] >= 5)

    p_email = raw.get("p_emaildomain") or ""
    r_email = raw.get("r_emaildomain") or ""
    features["p_email_is_high_risk"] = int(p_email in HIGH_RISK_EMAIL_DOMAINS)
    features["r_email_is_high_risk"] = int(r_email in HIGH_RISK_EMAIL_DOMAINS)
    features["p_email_is_known"] = int(p_email in KNOWN_EMAIL_DOMAINS)
    features["email_domain_match"] = int(p_email == r_email and p_email != "")

    features["product_is_high_risk"] = int(raw.get("product_cd", "") in HIGH_RISK_PRODUCT_CODES)
    features["is_credit"] = int(raw.get("card_type", "") == "credit")
    features["is_foreign"] = int(raw.get("is_foreign_transaction", False))
    features["is_mobile"] = int(raw.get("device_type", "") == "mobile")

    time_since = raw.get("time_since_last_txn_hours")
    features["time_since_last_txn"] = time_since if time_since is not None else -1
    features["rapid_repeat"] = int(time_since is not None and time_since < 0.1)

    # Return as ordered array matching training column order
    column_order = [
        "transaction_amt", "transaction_amt_log",
        "amt_vs_mean_ratio", "amt_deviation",
        "txn_count_1h", "txn_count_24h", "txn_velocity_ratio",
        "hour_of_day", "day_of_week", "is_night", "is_weekend",
        "p_email_is_high_risk", "r_email_is_high_risk", "p_email_is_known", "email_domain_match",
        "product_is_high_risk", "is_credit", "is_foreign", "is_mobile",
        "time_since_last_txn", "rapid_repeat",
    ]
    return np.array([features[c] for c in column_order], dtype=np.float32)
