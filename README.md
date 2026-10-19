# Vehicle Price Estimation Using Machine Learning (PKR)

**Author:** Muhammad Awais | **Background:** BS Computer Science  
**Focus:** Data science · supervised regression · data preprocessing · model evaluation

## Overview

A reproducible Python project for **training and evaluating a vehicle-price regression model** using labeled vehicle-listing data. The application predicts an *approximate listing price in Pakistani Rupees (PKR)* from vehicle details such as make, model, registration/manufacture year, mileage, fuel, transmission and engine capacity.

> **Status:** Code is provided; no trained model, dataset, evaluation results or live deployment are included. Train and validate with a real, appropriately licensed dataset before claiming accuracy. Predictions are for educational analysis, not appraisals.

## Why this revision?

The original prototype used a general-purpose Vision Transformer image classifier (`google/vit-base-patch16-224`) and matched its generic labels against multiple datasets. That classifier **cannot reliably identify exact vehicle make, model, year or condition**. The prototype also generated some prices from arbitrary category constants or image dimensions, which cannot substantiate a market valuation.

This revision uses a **genuinely trained supervised regression model** and a held-out test set instead. An optional image can be checked for validity, but **does not influence the predicted price**. The previous image-classification experiment can be described separately as exploratory work.

## Architecture

```text
Verified CSV vehicle listings (prices already in PKR)
              |
              v
   Validation + data cleaning (Pandas)
              |
              v
    Train / test split (80% / 20%)
              |
              v
Categorical imputation + OneHotEncoder
Numeric median imputation
              |
              v
  Random Forest Regressor (scikit-learn)
              |
       +------+------+
       |             |
  Evaluation     Saved model
  MAE/RMSE/R²    (.joblib)
                     |
                     v
            Predict from vehicle details
```

## Technology stack

Python, Pandas, NumPy, scikit-learn, joblib, Pillow (optional image validation).

## Repository files

| File | Purpose |
| --- | --- |
| `train.py` | Train, evaluate, save model and predict from vehicle details |
| `requirements.txt` | Python dependencies |
| `sample_data.csv` | Synthetic **format demonstration only**, not a training dataset |
| `README.md` | Documentation |

## Dataset requirements

Provide a CSV with **actual labeled listings**, using the following exact columns:

`make,model,year,mileage_km,fuel,transmission,engine_cc,price_pkr`

- Prices must already be **in PKR**. Do not mix currencies or silently assume exchange rates.
- Keep source, collection date, usage rights, and any cleaning/normalization steps documented.
- Include diverse vehicles and realistic listings. At least **30 valid rows** are required by the script; substantially more data is recommended.
- Check for duplicated listings, unrealistic prices and distribution shift. Do not use the included synthetic sample to report model accuracy.
- Source datasets from Hugging Face/Kaggle only after confirming schema, license, units and provenance. They are **not automatically merged** by this revision because their schemas/currencies may differ.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

### Train and evaluate

```bash
python train.py train --csv data/vehicle_listings_pkr.csv
```

The script performs an 80/20 random train/test split (seed 42), reports **MAE, RMSE and R²** on held-out data, and saves:

- `artifacts/vehicle_price_model.joblib`
- `artifacts/metrics.json`

These files are generated only **after successful training**.

### Predict

```bash
python train.py predict --make Toyota --car-model Corolla --year 2018 --mileage-km 75000 --fuel Petrol --transmission Automatic --engine-cc 1800
```

Optional image validation (does **not** change the price):

```bash
python train.py predict --make Toyota --car-model Corolla --year 2018 --mileage-km 75000 --fuel Petrol --transmission Automatic --engine-cc 1800 --image path/to/car.jpg
```

Only load trusted `.joblib` model files. Avoid committing private datasets, credentials or large model artifacts.

## Evaluation and limitations

- **MAE (PKR):** average absolute price error; lower is better.
- **RMSE (PKR):** penalizes larger errors more heavily.
- **R²:** goodness-of-fit on the held-out split; may be negative.
- The reported 90th-percentile absolute test error is **not** a calibrated confidence or prediction interval.
- Random splits can overestimate real-world generalization when repeated listings, same vehicles, or time drift exist. A temporal or grouped holdout is preferable for a serious study.
- No reliable price can be inferred from a photograph alone without vehicle identity, condition and local market context.
- Prices vary by region, date, condition, accident history and listing versus transaction value.

## Suggested next steps

1. Collect and document a sufficiently large, properly licensed PKR dataset.
2. Run training and save genuine metrics and plots (actual vs. predicted, residuals).
3. Compare against a simple baseline and evaluate a time-based test split.
4. Add a separate, verified make/model detector only if labeled automotive image data becomes available.
5. Add a Streamlit or other user interface once the prediction model is validated.

## Portfolio summary

This project demonstrates a reproducible **tabular machine-learning workflow** for vehicle listing price estimation, including data validation, categorical feature encoding, regression, held-out evaluation and inference. It does **not** claim production-grade valuation accuracy or image-based car identification.

**Author:** Muhammad Awais
