"""
Feature engineering logic — single source of truth for notebooks and API.
Notebooks import from here; api/main.py imports from here.
"""
import numpy as np

HIGH_RISK_EMAIL_DOMAINS = {"anonymous.com", "aim.com", "protonmail.com"}
HIGH_RISK_PRODUCT_CODES = {"H", "C"}

KNOWN_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "icloud.com", "aol.com", "comcast.net", "msn.com",
}

COLUMN_ORDER = [
    "transaction_amt", "transaction_amt_log",
    "amt_vs_mean_ratio", "amt_deviation",
    "txn_count_1h", "txn_count_24h", "txn_velocity_ratio",
    "hour_of_day", "day_of_week", "is_night", "is_weekend",
    "p_email_is_high_risk", "r_email_is_high_risk", "p_email_is_known", "email_domain_match",
    "product_is_high_risk", "is_credit", "is_foreign", "is_mobile",
    "time_since_last_txn", "rapid_repeat",
]


def engineer_features(raw: dict) -> np.ndarray:
    """Transform a raw transaction dict into the feature vector the model expects."""
    f = {}

    f["transaction_amt"] = raw["transaction_amt"]
    f["transaction_amt_log"] = np.log1p(raw["transaction_amt"])

    f["amt_vs_mean_ratio"] = raw.get("amt_vs_mean_ratio", 1.0)
    f["amt_deviation"] = np.log1p(
        abs(raw["transaction_amt"] * f["amt_vs_mean_ratio"] - raw["transaction_amt"])
    )

    f["txn_count_1h"] = raw.get("txn_count_1h", 0)
    f["txn_count_24h"] = raw.get("txn_count_24h", 0)
    f["txn_velocity_ratio"] = f["txn_count_1h"] / max(f["txn_count_24h"], 1)

    f["hour_of_day"] = raw["hour_of_day"]
    f["day_of_week"] = raw["day_of_week"]
    f["is_night"] = int(raw["hour_of_day"] < 6 or raw["hour_of_day"] >= 23)
    f["is_weekend"] = int(raw["day_of_week"] >= 5)

    p_email = raw.get("p_emaildomain") or ""
    r_email = raw.get("r_emaildomain") or ""
    f["p_email_is_high_risk"] = int(p_email in HIGH_RISK_EMAIL_DOMAINS)
    f["r_email_is_high_risk"] = int(r_email in HIGH_RISK_EMAIL_DOMAINS)
    f["p_email_is_known"] = int(p_email in KNOWN_EMAIL_DOMAINS)
    f["email_domain_match"] = int(p_email == r_email and p_email != "")

    f["product_is_high_risk"] = int(raw.get("product_cd", "") in HIGH_RISK_PRODUCT_CODES)
    f["is_credit"] = int(raw.get("card_type", "") == "credit")
    f["is_foreign"] = int(raw.get("is_foreign_transaction", False))
    f["is_mobile"] = int(raw.get("device_type", "") == "mobile")

    time_since = raw.get("time_since_last_txn_hours")
    f["time_since_last_txn"] = time_since if time_since is not None else -1
    f["rapid_repeat"] = int(time_since is not None and time_since < 0.1)

    return np.array([f[c] for c in COLUMN_ORDER], dtype=np.float32)
