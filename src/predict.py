"""
================================================================================
 FRAUD DETECTION PIPELINE — PREDICTION
================================================================================
 File        : predict.py
 Project     : Online Payment Fraud Detection
 Description : Loads the saved best model, preprocessor and feature list.
               Accepts a single transaction as input, preprocesses it,
               and returns fraud prediction with probability score.
               This file is used by app.py for live predictions.

 Imports from: config.py
 Used by     : app/app.py
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import joblib
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Import from config ─────────────────────────────────────────────────────────
from src.config import (
    BEST_MODEL_PATH,
    FEATURES_PATH,
)


# ==============================================================================
# STEP 1 — LOAD MODEL, PREPROCESSOR & FEATURES
# ==============================================================================

def load_model_and_preprocessor():
    """
    Load the saved best model, preprocessor and feature list from disk.
    Called once when app.py starts — not on every prediction.

    Returns:
        model        : Trained best model
        preprocessor : Fitted ColumnTransformer
        features     : List of feature column names
    """

    # ── Load best model ────────────────────────────────────────────────────────
    model = joblib.load(BEST_MODEL_PATH)

    # ── Load preprocessor ──────────────────────────────────────────────────────
    preprocessor_path = BEST_MODEL_PATH.parent / "preprocessor.joblib"
    preprocessor      = joblib.load(preprocessor_path)

    # ── Load feature list ──────────────────────────────────────────────────────
    # Ensures input data always has correct columns in correct order
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    print(f"Model loaded: {BEST_MODEL_PATH.name}")
    print(f"Preprocessor loaded: preprocessor.joblib")
    print(f"Features loaded: {len(features)} columns")

    return model, preprocessor, features


# ==============================================================================
# STEP 2 — VALIDATE INPUT
# ==============================================================================

def validate_input(input_data: dict, features: list) -> pd.DataFrame:
    """
    Validate and convert raw input dictionary into a DataFrame.
    Ensures all required features are present and in correct order.

    Args:
        input_data (dict) : Raw transaction data from app.py
        features   (list) : Expected feature names from feature_list.json

    Returns:
        pd.DataFrame: Single row DataFrame ready for preprocessing
    """

    # ── Check all required features are present ────────────────────────────────
    missing = [f for f in features if f not in input_data]
    if missing:
        raise ValueError(f"Missing features in input: {missing}")

    # ── Convert to DataFrame with correct column order ─────────────────────────
    df = pd.DataFrame([input_data])[features]

    return df


# ==============================================================================
# STEP 3 — PREDICT SINGLE TRANSACTION
# ==============================================================================

def predict_single(
    input_data   : dict,
    model        = None,
    preprocessor = None,
    features     : list = None,
) -> dict:
    """
    Predict fraud for a single transaction.

    Args:
        input_data   (dict) : Transaction features as key-value pairs
        model               : Loaded model (optional — loads if not provided)
        preprocessor        : Loaded preprocessor (optional)
        features     (list) : Feature names (optional)

    Returns:
        dict: {
            "prediction"  : "Fraud" or "Legitimate",
            "label"       : 1 or 0,
            "probability" : float (0.0 - 1.0),
            "confidence"  : "High" / "Medium" / "Low"
        }

    Example:
        >>> result = predict_single({
        ...     "account_age_days"    : 120,
        ...     "avg_monthly_spend"   : 5000.0,
        ...     "transaction_amount"  : 9500.0,
        ...     ...
        ... })
        >>> print(result)
        {"prediction": "Fraud", "label": 1, "probability": 0.94, "confidence": "High"}
    """

    # ── Load model if not provided ─────────────────────────────────────────────
    # This allows app.py to load once and reuse across predictions
    if model is None or preprocessor is None or features is None:
        model, preprocessor, features = load_model_and_preprocessor()

    # ── Validate and prepare input ─────────────────────────────────────────────
    input_df = validate_input(input_data, features)

    # ── Preprocess input ───────────────────────────────────────────────────────
    # Use transform only — never fit on new data
    input_processed = preprocessor.transform(input_df)

    # ── Predict ────────────────────────────────────────────────────────────────
    label       = int(model.predict(input_processed)[0])
    probability = float(model.predict_proba(input_processed)[0][1])

    # ── Confidence level based on probability ──────────────────────────────────
    if probability >= 0.80:
        confidence = "High"
    elif probability >= 0.50:
        confidence = "Medium"
    else:
        confidence = "Low"

    # ── Build result ───────────────────────────────────────────────────────────
    result = {
        "prediction" : "Fraud"      if label == 1 else "Legitimate",
        "label"      : label,
        "probability": round(probability, 4),
        "confidence" : confidence,
    }

    return result


# ==============================================================================
# SANITY CHECK — run: python -m src.predict
# ==============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("PREDICT.PY — SANITY CHECK")
    print("=" * 55)

    # ── Load model once ────────────────────────────────────────────────────────
    model, preprocessor, features = load_model_and_preprocessor()

    # ── Test 1 — Likely Legitimate transaction ─────────────────────────────────
    legit_transaction = {
        "account_age_days"                : 1200,
        "credit_score_band"               : 4,
        "kyc_level"                       : 3,
        "avg_monthly_spend"               : 4500.0,
        "merchant_risk_score"             : 0.05,
        "transaction_amount"              : 350.0,
        "payment_channel"                 : "upi",
        "device_type"                     : "mobile",
        "is_international"                : 0,
        "ip_risk_score"                   : 0.08,
        "txn_count_1h"                    : 1,
        "txn_count_24h"                   : 3,
        "failed_txn_count_24h"            : 0,
        "geo_distance_from_last_txn"      : 2.5,
        "amount_deviation_from_user_mean" : 50.0,
        "post_auth_risk_score"            : 0.10,
    }

    # ── Test 2 — Likely Fraud transaction ──────────────────────────────────────
    fraud_transaction = {
        "account_age_days"                : 15,
        "credit_score_band"               : 1,
        "kyc_level"                       : 1,
        "avg_monthly_spend"               : 1000.0,
        "merchant_risk_score"             : 0.92,
        "transaction_amount"              : 9800.0,
        "payment_channel"                 : "card",
        "device_type"                     : "desktop",
        "is_international"                : 1,
        "ip_risk_score"                   : 0.95,
        "txn_count_1h"                    : 8,
        "txn_count_24h"                   : 15,
        "failed_txn_count_24h"            : 4,
        "geo_distance_from_last_txn"      : 850.0,
        "amount_deviation_from_user_mean" : 8500.0,
        "post_auth_risk_score"            : 0.88,
    }

    # ── Run predictions ────────────────────────────────────────────────────────
    print("\nTest 1 — Legitimate Transaction:")
    result1 = predict_single(legit_transaction, model, preprocessor, features)
    print(f"Prediction  : {result1['prediction']}")
    print(f"Probability : {result1['probability']}")
    print(f"Confidence  : {result1['confidence']}")

    print("\nTest 2 — Fraud Transaction:")
    result2 = predict_single(fraud_transaction, model, preprocessor, features)
    print(f"Prediction  : {result2['prediction']}")
    print(f"Probability : {result2['probability']}")
    print(f"Confidence  : {result2['confidence']}")

    print("\n" + "=" * 55)
    print("predict.py working correctly")
    print("=" * 55)