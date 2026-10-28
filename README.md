# Credit Default Assessment System

A web-based credit-card default risk demo built with **FastAPI**, **scikit-learn**, and a simple HTML/CSS/JavaScript frontend. It compares Logistic Regression and Random Forest model probabilities and returns a risk score, risk level, explanation, and recommendation.

> **Disclaimer:** This is an educational project, not a validated credit-scoring product. The API accepts six simplified inputs and fills many model features with fixed defaults. Its risk score can therefore be misleading and must not be used for real lending or financial decisions.

## Features

- Web interface for entering customer details
- FastAPI prediction endpoint (`POST /predict`)
- Logistic Regression and Random Forest models
- Saved preprocessing/model artifacts in `models/`
- Health endpoint (`GET /health`)
- API documentation at `/docs`

## Repository structure

```text
credit-default-assessment-system/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── model.py
├── data/
│   └── UCI_Credit_Card.csv
├── models/
│   ├── feature_names.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── static/
│   ├── app.js
│   └── style.css
├── templates/
│   └── index.html
└── screenshots/
    └── (add application screenshots here)
```

## Requirements

- Python 3.10 or newer recommended
- pip

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-default-assessment-system.git
cd credit-default-assessment-system
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the API and web app

Run from the repository root:

```bash
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000> in your browser. Interactive API documentation is available at <http://127.0.0.1:8000/docs> and the health check at <http://127.0.0.1:8000/health>.

## API endpoint

`POST /predict`

Example request body:

```json
{
  "age": 30,
  "income": 50000,
  "credit_limit": 10000,
  "credit_utilization": 45,
  "late_payments": 1,
  "monthly_balance": 500
}
```

The `income` field is currently accepted by the API but the model's feature mapping does not use it to infer a true income-based relationship.

## Dataset

The included `data/UCI_Credit_Card.csv` is the UCI credit-card default dataset used by the training pipeline. Review the dataset's original terms and attribution before redistributing it or using it beyond educational purposes. The model can load the existing saved artifacts at startup; the dataset is needed if you retrain the models.

## Model artifacts

The saved `.pkl` files are required to start quickly without retraining. They are Python-specific serialized artifacts and should only be loaded from a trusted source. Re-training with `model.py` regenerates the files under `models/`.

## Limitations

- Several model features are not collected from the user and are filled with fixed defaults.
- The interface's six input fields do not represent all features used during training.
- Reported probabilities are not evidence of calibrated real-world default risk.
- The model and API have not been validated for real financial decisions.

## Suggested improvements

- Collect and validate all relevant model features, or retrain using only the fields actually collected.
- Add tests for preprocessing, prediction, and API validation.
- Record evaluation metrics on a held-out dataset and assess probability calibration.
- Add screenshots of the interface and example predictions to `screenshots/`.

## License

No project license is included. Add a license only after confirming that you have the rights to license your own code and that dataset/model redistribution terms permit it.
