"""
================================================================================
 FRAUD DETECTION PIPELINE — MODEL TRAINING
================================================================================
 File        : train.py
 Project     : Online Payment Fraud Detection
 Description : Loads preprocessed data, performs train/test split,
               fits preprocessor, trains all 5 models, evaluates them,
               compares results, saves best model and feature list,
               and exports predicted_output.csv.

 Imports from: config.py, preprocessing.py
 Output      : best_model.joblib, feature_list.json, predicted_output.csv
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import json
import joblib
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ── Import from config ─────────────────────────────────────────────────────────
from src.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    PRIMARY_METRIC,
    BEST_MODEL_PATH,
    FEATURES_PATH,
    OUTPUT_DIR,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    CATBOOST_PARAMS,
)

# ── Import from preprocessing ──────────────────────────────────────────────────
from src.preprocessing import run_preprocessing


# ==============================================================================
# STEP 1 — LOAD & PREPROCESS DATA
# ==============================================================================

def load_and_prepare_data():
    """
    Load cleaned dataset and build preprocessor via preprocessing.py.

    Returns:
        X            (pd.DataFrame)     : Full feature matrix
        y            (pd.Series)        : Target vector
        preprocessor (ColumnTransformer): Unfitted preprocessor
        df           (pd.DataFrame)     : Original full dataframe
                                          (kept for predicted_output.csv)
    """

    print("=" * 60)
    print("  STEP 1 — LOADING & PREPROCESSING DATA")
    print("=" * 60)

    # ── Run preprocessing pipeline ─────────────────────────────────────────────
    X, y, preprocessor = run_preprocessing()

    return X, y, preprocessor


# ==============================================================================
# STEP 2 — TRAIN / TEST SPLIT
# ==============================================================================

def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Split dataset into train and test sets.
    Stratified split preserves fraud/legit ratio in both sets.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series)   : Target vector

    Returns:
        X_train, X_test, y_train, y_test
    """

    print("\n" + "=" * 60)
    print("  STEP 2 — TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size   = TEST_SIZE,
        random_state= RANDOM_STATE,
        stratify    = y,           # preserves fraud ratio in both sets
    )

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train fraud rate : {y_train.mean()*100:.2f}%")
    print(f"y_test  fraud rate : {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# STEP 3 — FIT & TRANSFORM
# ==============================================================================

def fit_transform_data(preprocessor, X_train, X_test):
    """
    Fit preprocessor on X_train ONLY then transform both sets.
    Fitting on X_train only prevents data leakage from test set.

    Args:
        preprocessor : Unfitted ColumnTransformer
        X_train      : Training features
        X_test       : Test features

    Returns:
        X_train_processed, X_test_processed
    """

    print("\n" + "=" * 60)
    print("  STEP 3 — FIT & TRANSFORM")
    print("=" * 60)

    # ── Fit ONLY on training data — never on test data ─────────────────────────
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)     # transform only

    print(f"X_train processed shape : {X_train_processed.shape}")
    print(f"X_test  processed shape : {X_test_processed.shape}")

    # ── Save preprocessor ──────────────────────────────────────────────────────
    preprocessor_path = BEST_MODEL_PATH.parent / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved : {preprocessor_path}")

    return X_train_processed, X_test_processed


# ==============================================================================
# STEP 4 — DEFINE MODELS
# ==============================================================================

def get_models(scale_pos_weight: float) -> dict:
    """
    Initialize all 5 models with hyperparameters from config.py.
    scale_pos_weight is calculated dynamically for XGBoost.

    Args:
        scale_pos_weight (float): negatives / positives ratio

    Returns:
        dict: model name → model instance
    """

    print("\n" + "=" * 60)
    print("  STEP 4 — INITIALIZING MODELS")
    print("=" * 60)
    print(f"  XGBoost scale_pos_weight : {scale_pos_weight:.2f}")

    models = {
        # ── Baseline linear model ──────────────────────────────────────────────
        "Logistic Regression": LogisticRegression(
            **LOGISTIC_REGRESSION_PARAMS
        ),

        # ── Ensemble of decision trees ─────────────────────────────────────────
        "Random Forest": RandomForestClassifier(
            **RANDOM_FOREST_PARAMS
        ),

        # ── Gradient boosting — top performer on fraud data ────────────────────
        "XGBoost": XGBClassifier(
            **XGBOOST_PARAMS,
            scale_pos_weight=scale_pos_weight,  # set dynamically
        ),

        # ── Fast gradient boosting ─────────────────────────────────────────────
        "LightGBM": LGBMClassifier(
            **LIGHTGBM_PARAMS
        ),

        # ── Gradient boosting with native categorical support ──────────────────
        "CatBoost": CatBoostClassifier(
            **CATBOOST_PARAMS
        ),
    }

    for name in models:
        print(f"{name} initialized")

    return models


# ==============================================================================
# STEP 5 — TRAIN & EVALUATE ALL MODELS
# ==============================================================================

def train_and_evaluate(
    models      : dict,
    X_train     : np.ndarray,
    X_test      : np.ndarray,
    y_train     : pd.Series,
    y_test      : pd.Series,
) -> dict:
    """
    Train all models, evaluate on test set, and run cross validation.

    Args:
        models  : dict of model name → model instance
        X_train : Processed training features
        X_test  : Processed test features
        y_train : Training labels
        y_test  : Test labels

    Returns:
        dict: results per model
    """

    print("\n" + "=" * 60)
    print("  STEP 5 — TRAINING & EVALUATING ALL MODELS")
    print("=" * 60)

    results = {}
    skf     = StratifiedKFold(
        n_splits   = CV_FOLDS,
        shuffle    = True,
        random_state=RANDOM_STATE,
    )

    for name, model in models.items():

        print(f"\n  ── {name} ──────────────────────────────────")

        # ── Train ──────────────────────────────────────────────────────────────
        model.fit(X_train, y_train)

        # ── Predict ────────────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # ── Metrics ────────────────────────────────────────────────────────────
        roc_auc  = roc_auc_score(y_test, y_prob)
        pr_auc   = average_precision_score(y_test, y_prob)
        f1       = f1_score(y_test, y_pred)
        precision= precision_score(y_test, y_pred, zero_division=0)
        recall   = recall_score(y_test, y_pred)

        # ── Cross Validation ───────────────────────────────────────────────────
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv     = skf,
            scoring= "roc_auc",
            n_jobs = -1,
        )

        # ── Classification Report ──────────────────────────────────────────────
        print(classification_report(
            y_test, y_pred,
            target_names=["Legit", "Fraud"]
        ))
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(f"  PR-AUC   : {pr_auc:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── Store results ──────────────────────────────────────────────────────
        results[name] = {
            "model"    : model,
            "y_pred"   : y_pred,
            "y_prob"   : y_prob,
            "roc_auc"  : roc_auc,
            "pr_auc"   : pr_auc,
            "f1"       : f1,
            "precision": precision,
            "recall"   : recall,
            "cv_mean"  : cv_scores.mean(),
            "cv_std"   : cv_scores.std(),
        }

    return results


# ==============================================================================
# STEP 6 — MODEL COMPARISON TABLE
# ==============================================================================

def compare_models(results: dict) -> str:
    """
    Build and print a comparison table of all models.
    Selects best model based on PRIMARY_METRIC from config (PR-AUC).

    Args:
        results (dict): Training results per model

    Returns:
        str: Name of the best model
    """

    print("\n" + "=" * 60)
    print("  STEP 6 — MODEL COMPARISON TABLE")
    print("=" * 60)

    # ── Build comparison dataframe ─────────────────────────────────────────────
    comparison = pd.DataFrame([
        {
            "Model"    : name,
            "ROC-AUC"  : round(res["roc_auc"],   4),
            "PR-AUC"   : round(res["pr_auc"],    4),
            "F1"       : round(res["f1"],         4),
            "Precision": round(res["precision"],  4),
            "Recall"   : round(res["recall"],     4),
            "CV ROC-AUC": f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}",
        }
        for name, res in results.items()
    ])

    comparison = comparison.set_index("Model")
    print(f"\n{comparison.to_string()}")

    # ── Select best model by PR-AUC (primary metric for fraud) ────────────────
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"\nBest Model : {best_name}")
    print(f"PR-AUC    : {results[best_name]['pr_auc']:.4f}")
    print(f"ROC-AUC   : {results[best_name]['roc_auc']:.4f}")

    # ── Save comparison CSV ────────────────────────────────────────────────────
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path)
    print(f"\nComparison table saved : {comparison_path}")

    return best_name


# ==============================================================================
# STEP 7 — SAVE BEST MODEL & FEATURE LIST
# ==============================================================================

def save_best_model(best_name: str, results: dict, feature_names: list):
    """
    Save the best model as .joblib and feature names as .json.

    Args:
        best_name     (str)  : Name of the best model
        results       (dict) : All model results
        feature_names (list) : Feature column names used for training
    """

    print("\n" + "=" * 60)
    print("  STEP 7 — SAVING BEST MODEL")
    print("=" * 60)

    # ── Save best model ────────────────────────────────────────────────────────
    best_model = results[best_name]["model"]
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"Best model saved : {BEST_MODEL_PATH}")

    # ── Save feature list ──────────────────────────────────────────────────────
    # Critical — ensures predict.py and app.py use exact same features
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f, indent=4)
    print(f"Feature list saved : {FEATURES_PATH}")


# ==============================================================================
# STEP 8 — EXPORT PREDICTED OUTPUT CSV
# ==============================================================================

def export_predictions(
    best_name : str,
    results   : dict,
    X_test    : pd.DataFrame,
    y_test    : pd.Series,
):
    """
    Export predicted_output.csv with original features +
    predicted_label + fraud_probability for comparison.

    Args:
        best_name : Name of the best model
        results   : All model results
        X_test    : Original (unprocessed) test features
        y_test    : Actual test labels
    """

    print("\n" + "=" * 60)
    print("  STEP 8 — EXPORTING PREDICTIONS")
    print("=" * 60)

    # ── Build output dataframe ─────────────────────────────────────────────────
    output_df = X_test.copy()
    output_df[TARGET_COLUMN]        = y_test.values
    output_df["predicted_label"]    = results[best_name]["y_pred"]
    output_df["fraud_probability"]  = results[best_name]["y_prob"].round(4)

    # ── Add prediction result column ───────────────────────────────────────────
    output_df["result"] = output_df.apply(
        lambda row: "Correct" if row[TARGET_COLUMN] == row["predicted_label"]
                    else "Wrong", axis=1
    )

    # ── Save to outputs/ ───────────────────────────────────────────────────────
    output_path = OUTPUT_DIR / "predicted_output.csv"
    output_df.to_csv(output_path, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    correct = (output_df["result"] == "Correct").sum()
    total   = len(output_df)

    print(f"Predictions exported : {output_path}")
    print(f"Correct predictions  : {correct:,} / {total:,}")
    print(f"Accuracy             : {correct/total*100:.2f}%")


# ==============================================================================
# MAIN — RUN FULL TRAINING PIPELINE
# ==============================================================================

def main():
    """
    Orchestrates the full training pipeline:

        1. Load & preprocess data
        2. Train/test split
        3. Fit & transform
        4. Initialize models
        5. Train & evaluate all models
        6. Compare models
        7. Save best model & feature list
        8. Export predicted_output.csv
    """

    print("\nFRAUD DETECTION — TRAINING PIPELINE STARTING\n")

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    X, y, preprocessor = load_and_prepare_data()

    # ── Step 2: Split ──────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Step 3: Fit & Transform ────────────────────────────────────────────────
    X_train_p, X_test_p = fit_transform_data(preprocessor, X_train, X_test)

    # ── Step 4: Calculate scale_pos_weight for XGBoost ────────────────────────
    # Formula → total negatives / total positives
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # ── Step 5: Initialize models ──────────────────────────────────────────────
    models = get_models(scale_pos_weight)

    # ── Step 6: Train & evaluate ───────────────────────────────────────────────
    results = train_and_evaluate(
        models, X_train_p, X_test_p, y_train, y_test
    )

    # ── Step 7: Compare models ─────────────────────────────────────────────────
    best_name = compare_models(results)

    # ── Step 8: Save best model + feature list ─────────────────────────────────
    save_best_model(best_name, results, list(X.columns))

    # ── Step 9: Export predictions ─────────────────────────────────────────────
    export_predictions(best_name, results, X_test, y_test)

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print(f"Best Model : {best_name}")
    print(f"Model saved: {BEST_MODEL_PATH}")
    print(f"Output dir : {OUTPUT_DIR}")
    print("=" * 60)


# ==============================================================================
# SANITY CHECK — run: python src/train.py
# ==============================================================================

if __name__ == "__main__":
    main()