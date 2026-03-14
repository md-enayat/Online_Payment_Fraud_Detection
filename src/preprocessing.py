"""
================================================================================
 FRAUD DETECTION PIPELINE — PREPROCESSING
================================================================================
 File        : preprocessing.py
 Project     : Online Payment Fraud Detection
 Description : Loads cleaned dataset, separates features and target,
               auto-detects column types, builds ColumnTransformer
               pipeline. Train/test split and fitting happens in train.py.
 Returns     : X, y, preprocessor
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

#--- Import from config --------------------------------------------------------
from src.config import DATA_FILE, TARGET_COLUMN


# ==============================================================================
# FUNCTION 1 — LOAD DATA
# ==============================================================================

def data_load(data_path: Path = DATA_FILE) -> pd.DataFrame:
    """
    Load the cleaned dataset from the specified path.

    Args:
        data_path (Path): Path to the cleaned CSV file.

    Returns:
        pd.DataFrame: Cleaned dataset loaded from CSV.
    """
    df = pd.read_csv(data_path)

    # Normalize column names — strip whitespace
    df.columns = [str(c).strip() for c in df.columns]

    print(f"Dataset loaded successfully")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    return df


# ==============================================================================
# FUNCTION 2 — SEPARATE FEATURES AND TARGET
# ==============================================================================

def separate_features_target(df: pd.DataFrame):
    """
    Separate dataset into features (X) and target (y).
    Drops transaction_id — identifier, not a feature.

    Args:
        df (pd.DataFrame): Full cleaned dataset.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series)   : Target (is_fraud).
    """

    # ── Drop transaction_id and target ─────────────────────────────────────────
    X = df.drop(columns=["transaction_id", TARGET_COLUMN])

    # ── Target vector ──────────────────────────────────────────────────────────
    y = df[TARGET_COLUMN]

    print(f"\n Features shape : {X.shape}")
    print(f"Target shape   : {y.shape}")

    return X, y


# ==============================================================================
# FUNCTION 3 — BUILD PREPROCESSOR
# ==============================================================================

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a reusable preprocessing pipeline for numerical
    and categorical features.

    Args:
        X (pd.DataFrame): Feature matrix.

    Returns:
        ColumnTransformer: Unfitted preprocessor — fitted in train.py
                           on X_train only to prevent data leakage.
    """

    # ==========================================================================
    # Identify numerical and categorical columns
    # ==========================================================================

    # Detect numerical columns
    num_cols = X.select_dtypes(
        include=["int64", "float64", "number"]
    ).columns.tolist()

    # Detect categorical columns
    cat_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"\nNumerical columns   ({len(num_cols)}) : {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}) : {cat_cols}")

    # ==========================================================================
    # Numerical Pipeline
    # ==========================================================================
    # median   → robust to outliers in fraud transaction data
    # Scaler   → required for Logistic Regression to converge

    numerical_pipeline = Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy="median")),  # median 
        ("Scaler" , StandardScaler()),
    ])

    # ==========================================================================
    # Categorical Pipeline
    # ==========================================================================
    # most_frequent    → safest fill for missing categories
    # sparse_output    → False required by XGBoost, LightGBM, CatBoost

    categorical_pipeline = Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot" , OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,          # False 
        )),
    ])

    # ==========================================================================
    # Combine into ColumnTransformer
    # ==========================================================================

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline,    num_cols),
        ("cat", categorical_pipeline,  cat_cols),
    ])

    print(f"\n Preprocessor built successfully")
    print(f"Numerical  → SimpleImputer(median) + StandardScaler")
    print(f"Categorical → SimpleImputer(mode) + OneHotEncoder")
    print(f"Note: Preprocessor is unfitted — fitting happens in train.py")

    return preprocessor


# ==============================================================================
# MAIN — RUN PREPROCESSING
# ==============================================================================

def run_preprocessing():
    """
    Orchestrates full preprocessing pipeline.

    Returns:
        X            (pd.DataFrame)    : Feature matrix
        y            (pd.Series)       : Target vector
        preprocessor (ColumnTransformer): Unfitted transformer
    """

    # ── Load ───────────────────────────────────────────────────────────────────
    df = data_load()

    # ── Separate X and y ───────────────────────────────────────────────────────
    X, y = separate_features_target(df)

    # ── Build preprocessor ─────────────────────────────────────────────────────
    preprocessor = build_preprocessor(X)

   
    print(f"PREPROCESSING COMPLETE")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Preprocessor: unfitted — ready for training.py")
   

    return X, y, preprocessor


# ==============================================================================
# SANITY CHECK — run: python src/preprocessing.py
# ==============================================================================

if __name__ == "__main__":
    X, y, preprocessor = run_preprocessing()