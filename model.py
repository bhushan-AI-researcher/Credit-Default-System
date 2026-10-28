"""
model.py — Credit Default Prediction ML Pipeline
Handles: preprocessing, feature engineering, training, and inference.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "UCI_Credit_Card.csv")  # dataset stored in data/
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ─── 1. Load & Preprocess ─────────────────────────────────────────────────────
def load_and_preprocess(data_path: str = DATA_PATH):
    """Load the UCI Credit-Card dataset, clean it, and engineer features."""
    df = pd.read_csv(data_path)

    # Drop ID column (not a feature)
    df.drop(columns=["ID"], errors="ignore", inplace=True)

    # Target column (rename for convenience)
    target_col = "default.payment.next.month"

    # ── Handle known bad categorical values ────────────────────────────────
    # EDUCATION: values 0, 5, 6 are undocumented → map them to "Other" (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # MARRIAGE: value 0 is undocumented → "Other" (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # ── Feature Engineering ────────────────────────────────────────────────
    # Credit-utilization ratio: latest bill vs. credit limit
    df["credit_utilization_ratio"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1e-9)

    # Average monthly payment over 6 months
    pay_amt_cols  = [f"PAY_AMT{i}" for i in range(1, 7)]
    bill_amt_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    df["avg_pay_amt"]  = df[pay_amt_cols].mean(axis=1)
    df["avg_bill_amt"] = df[bill_amt_cols].mean(axis=1)

    # Debt-to-income ratio (proxy: avg bill / credit limit)
    df["debt_to_income_ratio"] = df["avg_bill_amt"] / (df["LIMIT_BAL"] + 1e-9)

    # Count months with late payments (PAY_X > 0 means delayed payment)
    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["late_payment_count"] = (df[pay_status_cols] > 0).sum(axis=1)

    # ── Split features / target ────────────────────────────────────────────
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y, df


# ─── 2. Train models ──────────────────────────────────────────────────────────
def train_models():
    """Train Logistic Regression & Random Forest; save artefacts to disk."""
    print("=" * 60)
    print("  Credit Default Prediction — Model Training")
    print("=" * 60)

    X, y, _ = load_and_preprocess()

    # Train / test split (stratified to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Scale features ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Logistic Regression ────────────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_sc, y_train)
    lr_pred  = lr.predict(X_test_sc)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]

    print("\n📌 Logistic Regression")
    print(f"   Accuracy : {accuracy_score(y_test, lr_pred):.4f}")
    print(f"   ROC-AUC  : {roc_auc_score(y_test, lr_proba):.4f}")
    print(classification_report(y_test, lr_pred, target_names=["No Default", "Default"]))

    # ── Random Forest ──────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    rf.fit(X_train, y_train)           # RF doesn't need scaled features
    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    print("\n📌 Random Forest")
    print(f"   Accuracy : {accuracy_score(y_test, rf_pred):.4f}")
    print(f"   ROC-AUC  : {roc_auc_score(y_test, rf_proba):.4f}")
    print(classification_report(y_test, rf_pred, target_names=["No Default", "Default"]))

    # ── Feature importance ─────────────────────────────────────────────────
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top10 = importances.nlargest(10)
    print("\n🌲 Top-10 Feature Importances (Random Forest):")
    for feat, imp in top10.items():
        bar = "█" * int(imp * 200)
        print(f"   {feat:<35} {imp:.4f}  {bar}")

    # ── Save artefacts ─────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(lr,     os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(rf,     os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_names.pkl"))

    print("\n✅ Models saved to ./models/")
    print("=" * 60)

    return {
        "lr_accuracy": float(accuracy_score(y_test, lr_pred)),
        "lr_roc_auc":  float(roc_auc_score(y_test, lr_proba)),
        "rf_accuracy": float(accuracy_score(y_test, rf_pred)),
        "rf_roc_auc":  float(roc_auc_score(y_test, rf_proba)),
        "feature_importances": top10.to_dict(),
        "feature_names": list(X.columns),
    }


# ─── 3. Load saved models ─────────────────────────────────────────────────────
def load_models():
    """Load pre-trained models from disk (called at API startup)."""
    lr     = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    rf     = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feats  = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return lr, rf, scaler, feats


# ─── 4. Build inference row ────────────────────────────────────────────────────
def build_input_row(
    age: int,
    income: float,
    credit_limit: float,
    credit_utilization: float,    # 0-100 %
    late_payments: int,
    monthly_balance: float,
    feature_names: list,
) -> pd.DataFrame:
    """
    Map the 6 UI fields onto the full feature vector expected by the model.
    Fields not collected from the user are filled with sensible defaults
    (dataset column medians — stored inline to avoid a runtime dataset read).
    """
    # Convert UI percentage to the same scale as training (ratio)
    util_amount = (credit_utilization / 100.0) * credit_limit

    row = {feat: 0 for feat in feature_names}   # default everything to 0

    # Directly observable fields
    row["LIMIT_BAL"] = credit_limit
    row["AGE"]       = age
    row["BILL_AMT1"] = util_amount
    row["BILL_AMT2"] = util_amount
    row["BILL_AMT3"] = util_amount
    row["BILL_AMT4"] = util_amount
    row["BILL_AMT5"] = util_amount
    row["BILL_AMT6"] = util_amount

    # Monthly payments ≈ monthly_balance provided by user
    for i in range(1, 7):
        row[f"PAY_AMT{i}"] = monthly_balance

    # Late payment status: spread 'late_payments' across PAY_0 … PAY_5
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    for i, pc in enumerate(pay_cols):
        row[pc] = 1 if i < late_payments else 0   # 1 = 1-month delay

    # SEX, EDUCATION, MARRIAGE — neutral defaults
    row["SEX"]       = 1
    row["EDUCATION"] = 2   # university
    row["MARRIAGE"]  = 2   # single

    # Engineered features (must match training)
    avg_bill = util_amount
    row["credit_utilization_ratio"] = credit_utilization / 100.0
    row["avg_pay_amt"]              = monthly_balance
    row["avg_bill_amt"]             = avg_bill
    row["debt_to_income_ratio"]     = avg_bill / (credit_limit + 1e-9)
    row["late_payment_count"]       = late_payments

    return pd.DataFrame([row], columns=feature_names)


# ─── 5. Explain & Recommend ───────────────────────────────────────────────────
def generate_explanation(
    credit_utilization: float,
    late_payments: int,
    income: float,
    monthly_balance: float,
    credit_limit: float,
    risk_score: float,
) -> str:
    """Generate a human-readable risk explanation."""
    reasons = []

    if credit_utilization > 75:
        reasons.append("very high credit utilization (>75%), indicating heavy reliance on credit")
    elif credit_utilization > 50:
        reasons.append("elevated credit utilization (>50%)")

    if late_payments >= 4:
        reasons.append("multiple late payments (≥4 months), signaling significant financial stress")
    elif late_payments >= 2:
        reasons.append("several late payments, suggesting repayment difficulties")
    elif late_payments == 1:
        reasons.append("a recent late payment")

    util_amount = (credit_utilization / 100.0) * credit_limit
    if income > 0 and (util_amount / (income + 1e-9)) > 0.5:
        reasons.append("a high outstanding balance relative to income")

    if monthly_balance < util_amount * 0.05 and util_amount > 0:
        reasons.append("monthly payments well below the outstanding balance")

    if not reasons:
        reasons.append("stable financial metrics with low credit utilization and timely payments")

    risk_label = (
        "high risk" if risk_score >= 0.6
        else "medium risk" if risk_score >= 0.35
        else "low risk"
    )

    prefix = f"The customer is classified as {risk_label} (score: {risk_score:.0%})"
    if len(reasons) == 1 and "stable" in reasons[0]:
        return f"{prefix}. The profile shows {reasons[0]}."
    return f"{prefix} due to: {'; '.join(reasons)}."


def generate_recommendation(risk_level: str) -> str:
    """Return actionable business recommendation based on risk level."""
    recs = {
        "High": (
            "⚠️  Immediate action required: "
            "Consider reducing the credit limit by 20–30%, trigger risk alert workflow, "
            "send proactive communication to the customer, and flag for manual review "
            "by the credit risk team."
        ),
        "Medium": (
            "🔍  Enhanced monitoring: "
            "Set automated balance & payment alerts, schedule a quarterly review, "
            "and limit any further credit limit increases until payment behaviour improves."
        ),
        "Low": (
            "✅  Reward good standing: "
            "Consider offering a credit limit increase or loyalty rewards, "
            "promote premium card upgrade, and maintain standard monitoring cadence."
        ),
    }
    return recs.get(risk_level, "No recommendation available.")


# ─── 6. Predict ───────────────────────────────────────────────────────────────
def predict(
    lr, rf, scaler, feature_names,
    age, income, credit_limit,
    credit_utilization, late_payments, monthly_balance
) -> dict:
    """Run inference and return a structured response dict."""

    input_df = build_input_row(
        age, income, credit_limit,
        credit_utilization, late_payments,
        monthly_balance, feature_names,
    )

    # Use Random Forest probability as primary score (more accurate)
    input_sc    = scaler.transform(input_df)
    rf_prob     = float(rf.predict_proba(input_df)[0, 1])
    lr_prob     = float(lr.predict_proba(input_sc)[0, 1])

    # Ensemble: weighted average (RF is more reliable on tabular data)
    risk_score  = 0.7 * rf_prob + 0.3 * lr_prob

    # Risk level thresholds
    if risk_score >= 0.6:
        risk_level = "High"
    elif risk_score >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    explanation    = generate_explanation(
        credit_utilization, late_payments, income,
        monthly_balance, credit_limit, risk_score,
    )
    recommendation = generate_recommendation(risk_level)

    return {
        "risk_score":      round(risk_score * 100, 2),   # as %
        "risk_level":      risk_level,
        "explanation":     explanation,
        "recommendation":  recommendation,
        "lr_probability":  round(lr_prob * 100, 2),
        "rf_probability":  round(rf_prob * 100, 2),
    }


# ─── Entrypoint: train if run directly ────────────────────────────────────────
if __name__ == "__main__":
    train_models()
