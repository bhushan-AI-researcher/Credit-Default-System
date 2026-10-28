"""
main.py — Credit Default Prediction API
FastAPI application: serves the HTML frontend and exposes /predict endpoint.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from model import train_models, load_models, predict as ml_predict

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Global model state ───────────────────────────────────────────────────────
STATE: dict = {}


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train (or load) models at startup."""
    models_dir = os.path.join(BASE_DIR, "models")
    lr_path    = os.path.join(models_dir, "logistic_regression.pkl")

    if os.path.exists(lr_path):
        log.info("✅  Loading pre-trained models from disk …")
        lr, rf, scaler, feats = load_models()
    else:
        log.info("🚀  No saved models found — training now …")
        metrics = train_models()
        log.info(
            "📊  Training complete  |  LR accuracy=%.4f  RF accuracy=%.4f",
            metrics["lr_accuracy"], metrics["rf_accuracy"],
        )
        log.info(
            "📊  ROC-AUC            |  LR=%.4f  RF=%.4f",
            metrics["lr_roc_auc"], metrics["rf_roc_auc"],
        )
        log.info("🌲  Top features: %s", list(metrics["feature_importances"].keys())[:5])
        lr, rf, scaler, feats = load_models()

    STATE["lr"]     = lr
    STATE["rf"]     = rf
    STATE["scaler"] = scaler
    STATE["feats"]  = feats
    log.info("🟢  API ready")
    yield
    log.info("🔴  Shutting down")


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Default Prediction API",
    description="Predict credit card default risk using Logistic Regression & Random Forest.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend on any origin (development convenience)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS, JS)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static",
)


# ─── Request / Response schemas ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    age:               int   = Field(..., ge=18, le=100,  description="Customer age (18–100)")
    income:            float = Field(..., ge=0,            description="Annual income (USD)")
    credit_limit:      float = Field(..., ge=1,            description="Credit card limit (USD)")
    credit_utilization:float = Field(..., ge=0,  le=100,  description="Credit utilization (%)")
    late_payments:     int   = Field(..., ge=0,  le=6,    description="No. of late payments (0–6)")
    monthly_balance:   float = Field(..., ge=0,            description="Average monthly payment (USD)")

    @validator("income", "credit_limit", "monthly_balance", pre=True)
    def must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("Value must be ≥ 0")
        return v


class PredictResponse(BaseModel):
    risk_score:      float
    risk_level:      str
    explanation:     str
    recommendation:  str
    lr_probability:  float
    rf_probability:  float


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main HTML page."""
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))


@app.post("/predict", response_model=PredictResponse)
async def predict_default(req: PredictRequest):
    """
    Run credit-default prediction.

    Returns risk score (0–100 %), risk level, explanation, and recommendation.
    """
    if not STATE:
        raise HTTPException(status_code=503, detail="Models not yet loaded")

    log.info(
        "📥  Request  age=%d  income=%.0f  limit=%.0f  util=%.1f%%  late=%d  balance=%.0f",
        req.age, req.income, req.credit_limit,
        req.credit_utilization, req.late_payments, req.monthly_balance,
    )

    result = ml_predict(
        STATE["lr"], STATE["rf"], STATE["scaler"], STATE["feats"],
        req.age, req.income, req.credit_limit,
        req.credit_utilization, req.late_payments, req.monthly_balance,
    )

    log.info(
        "📤  Response risk_score=%.2f%%  risk_level=%s",
        result["risk_score"], result["risk_level"],
    )
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": bool(STATE)}
