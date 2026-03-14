# 🔍 Online Payment Fraud Detection
A production-ready machine learning pipeline for detecting fraudulent 
online payment transactions in real time.

---

## 🌐 Live Demo

👉 [Click here to try the app](https://your-app-link.streamlit.app)

---

## 📌 Project Overview

Online payment fraud is a growing threat in the digital economy.
This project builds an end-to-end ML pipeline that:

- Cleans and analyzes real-world transaction data
- Trains 5 machine learning models
- Selects the best model automatically
- Serves live predictions via a Streamlit web app

---

## 📊 Dataset

| Detail | Value |
|--------|-------|
| Source | Digital Payment Fraud Detection Benchmark Dataset |
| Rows | 99,887 transactions |
| Columns | 18 features |
| Target | `is_fraud` (0 = Legitimate, 1 = Fraud) |
| Fraud Rate | 1.99% — highly imbalanced |

### Features Used

| Feature | Description |
|---------|-------------|
| `transaction_amount` | Value of the transaction |
| `account_age_days` | Age of the account in days |
| `ip_risk_score` | Risk score of originating IP |
| `merchant_risk_score` | Pre-computed merchant risk |
| `geo_distance_from_last_txn` | Distance from last transaction |
| `amount_deviation_from_user_mean` | Deviation from user average |
| `txn_count_1h` | Transactions in last 1 hour |
| `failed_txn_count_24h` | Failed transactions in 24 hours |
| `post_auth_risk_score` | Post-authorization risk score |
| `payment_channel` | CARD / UPI / WALLET / NETBANKING |
| `device_type` | MOBILE / DESKTOP / TABLET |
| `credit_score_band` | Credit score category |
| `kyc_level` | KYC verification level |
| `is_international` | International transaction flag |

---

## 🏗️ Project Structure
```
Online_Payment_Fraud_Detection/
│
├── 📁 app/
│   └── app.py                  ← Streamlit web application
│
├── 📁 src/
│   ├── config.py               ← Central configuration
│   ├── preprocessing.py        ← Data preprocessing pipeline
│   ├── training.py             ← Model training & evaluation
│   └── predict.py              ← Single transaction prediction
│
├── 📁 notebook/
│   ├── Cleaning_Data.ipynb     ← Data cleaning steps
│   └── EDA.ipynb               ← Exploratory data analysis
│
├── 📁 data/
│   ├── cleaned_data.csv        ← Cleaned dataset
│   └── predicted_output.csv    ← Model predictions
│
├── 📁 models/
│   ├── best_model.joblib       ← Saved best model
│   ├── preprocessor.joblib     ← Saved preprocessor
│   └── feature_list.json       ← Feature names
│
├── 📁 outputs/
│   └── model_comparison.csv    ← Model comparison results
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🤖 Models Trained

| Model | Handles Imbalance |
|-------|-------------------|
| Logistic Regression | `class_weight=balanced` |
| Random Forest | `class_weight=balanced` |
| XGBoost | `scale_pos_weight` (dynamic) |
| LightGBM | `is_unbalance=True` |
| CatBoost | `auto_class_weights=Balanced` |

> Best model selected automatically based on **PR-AUC** — 
> the most meaningful metric for imbalanced fraud datasets.

---

## 📈 Evaluation Metrics

| Metric | Why Used |
|--------|----------|
| PR-AUC | Primary — best for imbalanced data |
| ROC-AUC | Secondary — overall discrimination |
| F1 Score | Balance between precision & recall |
| Precision | Of predicted frauds, how many are real |
| Recall | Of real frauds, how many were caught |

---

## ⚙️ Pipeline Architecture
```
Raw Data (99,887 rows)
        ↓
Data Cleaning          → removes irrelevant columns
        ↓
EDA                    → insights & visualizations
        ↓
Preprocessing          → imputation + scaling + encoding
        ↓
Train/Test Split       → 80/20 stratified
        ↓
Model Training         → 5 models + cross validation
        ↓
Best Model Selection   → by PR-AUC
        ↓
Prediction             → single transaction inference
        ↓
Streamlit App          → live fraud detection UI
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Online_Payment_Fraud_Detection.git
cd Online_Payment_Fraud_Detection
```

### 2. Create Environment
```bash
conda create -n venv310 python=3.10
conda activate venv310
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Training Pipeline
```bash
python -m src.training
```

### 5. Run Streamlit App
```bash
streamlit run app/app.py
```

---

## 🖥️ App Features

- 📋 **Transaction Input Form** — fill in all transaction details
- 📊 **Fraud Probability Gauge** — visual probability meter
- 🎯 **Confidence Level** — High / Medium / Low
- ⚠️ **Risk Factor Analysis** — explains why transaction is flagged
- 📈 **Transaction Summary** — key details at a glance

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Data | pandas, numpy |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost |
| Visualization | matplotlib, seaborn, plotly |
| App | Streamlit |
| Serialization | joblib |
| Version Control | Git, GitHub |

---

## 👨‍💻 Author

**Md Enayat**

---
