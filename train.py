"""Project 2: reproducible tabular vehicle-price model with optional image inspection.

This program does NOT infer exact make/model/year from a generic image classifier.
Train on verified vehicle listings with known price currency and features.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["make", "model", "fuel", "transmission"]
NUMERIC = ["year", "mileage_km", "engine_cc"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price_pkr"


def load_clean_data(csv_path):
    data = pd.read_csv(csv_path)
    missing = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
    data = data[FEATURES + [TARGET]].copy()
    for col in NUMERIC + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in CATEGORICAL:
        data[col] = data[col].astype("string").str.strip().replace("", pd.NA)
    data = data.dropna(subset=[TARGET])
    data = data[(data[TARGET] > 0) & (data["year"].between(1980, 2030)) &
                (data["mileage_km"].between(0, 2_000_000)) &
                (data["engine_cc"].between(300, 10000))]
    data = data.drop_duplicates()
    if len(data) < 30:
        raise ValueError("At least 30 valid rows are required; 100+ diverse listings recommended.")
    return data


def make_pipeline():
    preprocessing = ColumnTransformer([
        ("categories", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
        ("numbers", SimpleImputer(strategy="median"), NUMERIC),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                             random_state=42, n_jobs=-1)),
    ])


def train(args):
    data = load_clean_data(args.csv)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=42)
    model = make_pipeline()
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    errors = np.abs(y_test.to_numpy() - predicted)
    metrics = {
        "valid_rows": int(len(data)), "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)), "mae_pkr": round(float(mean_absolute_error(y_test, predicted)), 2),
        "rmse_pkr": round(float(np.sqrt(mean_squared_error(y_test, predicted))), 2),
        "r2": round(float(r2_score(y_test, predicted)), 4),
        "holdout_absolute_error_p90_pkr": round(float(np.percentile(errors, 90)), 2),
        "note": "Random holdout metrics are exploratory, not a guarantee of real-market accuracy.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, out)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {out}; metrics: {args.metrics}")


def predict(args):
    artifact = joblib.load(args.model)  # Only load models you created/trust.
    row = {"make": args.make, "model": args.car_model, "year": args.year,
           "mileage_km": args.mileage_km, "fuel": args.fuel,
           "transmission": args.transmission, "engine_cc": args.engine_cc}
    prediction = max(0.0, float(artifact["model"].predict(pd.DataFrame([row]))[0]))
    print(f"Estimated listing price: PKR {prediction:,.0f}")
    error = artifact["metrics"].get("holdout_absolute_error_p90_pkr")
    if error is not None:
        print(f"Illustrative ±90th-percentile holdout error: PKR {error:,.0f} (not a calibrated prediction interval)")
    if args.image:
        from PIL import Image
        with Image.open(args.image) as im:
            im.verify()
        print(f"Image validated: {args.image}. Image is NOT used to predict price or vehicle identity.")
    print("Research/demo estimate only; check comparable current local listings.")


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate or predict vehicle prices in PKR")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Train with labeled PKR listing data")
    t.add_argument("--csv", required=True)
    t.add_argument("--output", default="artifacts/vehicle_price_model.joblib")
    t.add_argument("--metrics", default="artifacts/metrics.json")
    t.set_defaults(func=train)
    p = sub.add_parser("predict", help="Predict from structured vehicle details")
    p.add_argument("--model", default="artifacts/vehicle_price_model.joblib")
    for name in ("make", "car-model", "fuel", "transmission"):
        p.add_argument("--" + name, required=True)
    for name in ("year", "mileage-km", "engine-cc"):
        p.add_argument("--" + name, required=True, type=int)
    p.add_argument("--image", help="Optional image validation only; not an input to price model")
    p.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
