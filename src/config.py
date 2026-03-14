"""
==========================================================================================================
FRAUD DETECTION PIPELINE - CONFIGURATION FILE
==========================================================================================================
File    : Config.py
Project : Online Payment Fraud Detection
Description : This is the SINGLE SOURCE OF TRUTH for the entire pipeline.
              All paths, hyperparameters, and settings live here.
              No other file should hardcode any value — everything
              must be imported from this config.
"""
from pathlib import Path

# ==============================================================================
# PROJECT PATHS
# ==============================================================================

#--- Project root --------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parents[1]

#--- Data ----------------------------------------------------------------------
DATA_DIR : Path = BASE_DIR / "data"
DATA_FILE: Path = DATA_DIR  / "cleaned_data.csv"

#--- Models ----------------------------------------------------------------------
MODEL_DIR      : Path = BASE_DIR / "models"
BEST_MODEL_PATH: Path = MODEL_DIR / "best_model.joblib" 
FEATURES_PATH  : Path = MODEL_DIR / "feature_list.json"  

#--- Outputs (plots, reports) ----------------------------------------------------
OUTPUT_DIR: Path = BASE_DIR / "outputs"

# ── Auto-create folders if they don't exist -------------------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# TARGET COLUMN
# ==============================================================================

# Label we are predicting (0 = Legitimate, 1 = Fraud)
TARGET_COLUMN: str = "is_fraud"


# ==============================================================================
# TRAIN / TEST SPLIT
# ==============================================================================

TEST_SIZE   : float = 0.2   # 80% train — 20% test
RANDOM_STATE: int   = 42    # fixed seed for reproducibility


# ==============================================================================
# MODEL HYPERPARAMETERS
# ==============================================================================

#--- Logistic Regression -------------------------------------------------------
LOGISTIC_REGRESSION_PARAMS: dict = {
    "max_iter"    : 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

#--- Random Forest------------------------------------------------------------------
RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth"   : 10,
    "class_weight": "balanced",
    "n_jobs"      : -1,
    "random_state": RANDOM_STATE,
}

#--- XGBoost ----------------------------------------------------------------------
# scale_pos_weight → set dynamically in train.py as (negatives / positives)
XGBOOST_PARAMS: dict = {
    "n_estimators" : 300,
    "max_depth"    : 6,
    "learning_rate": 0.05,
    "eval_metric"  : "aucpr",
    "n_jobs"       : -1,
    "random_state" : RANDOM_STATE,
}

#--- LightGBM ----------------------------------------------------------------------
LIGHTGBM_PARAMS: dict = {
    "n_estimators" : 300,
    "max_depth"    : 6,
    "learning_rate": 0.05,
    "is_unbalance" : True,
    "n_jobs"       : -1,
    "random_state" : RANDOM_STATE,
    "verbose"      : -1,
}

#--- CatBoost ----------------------------------------------------------------------
# Handles categorical features natively — no OneHotEncoding needed
CATBOOST_PARAMS: dict = {
    "iterations"  : 300,
    "depth"       : 6,
    "learning_rate": 0.05,
    "auto_class_weights": "Balanced",  # handles fraud imbalance
    "verbose"     : 0,                 # suppress training logs
    "random_seed" : RANDOM_STATE,
}

# ==============================================================================
# CROSS VALIDATION
# ==============================================================================

CV_FOLDS  : int  = 5
N_JOBS    : int  = -1
CV_SCORING: list = [
    "roc_auc",           # overall discrimination ability
    "average_precision", # PR-AUC — better for imbalanced fraud data
    "f1",                # balance between precision and recall
]


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

# PR-AUC is primary — more meaningful than ROC-AUC for imbalanced data
PRIMARY_METRIC  : str = "pr_auc"
SECONDARY_METRIC: str = "roc_auc"


# ==============================================================================
# PLOT SETTINGS
# ==============================================================================

PLOT_STYLE  : str   = "whitegrid"
PLOT_DPI    : int   = 150
FIGURE_SIZE : tuple = (12, 6)


# ==============================================================================
# run: python src/config.py
# ==============================================================================

print(f"Config loaded successfully")
print(f"Data      : {DATA_FILE}")
print(f"Models    : {MODEL_DIR}")
print(f"Outputs   : {OUTPUT_DIR}")
print(f"Target    : {TARGET_COLUMN}")