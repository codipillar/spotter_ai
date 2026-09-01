# Freight Rate Prediction Challenge

Machine learning take-home for Spotter: predict `posted_rate` for truckload freight.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

| File | Role |
|------|------|
| `data/train_test.csv` | Labeled development data (Jan–Oct 2025) |
| `data/validation.csv` | 12,000 loads to score (Nov–Dec 2025) |
| `data/validation_predictions_template.csv` | Submission template |
| `data/december_chart_inputs.csv` | Fixed Lexington → Fort Wayne December scenario |

## Reproduce predictions

```bash
cd src
python train_predict.py
cd ..
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

Outputs:

- `validation_predictions.csv`
- filled `data/december_chart_inputs.csv`
- `scorer_results/candidate_december.png`
- `artifacts/metrics.json` (holdout metrics + feature importance)

## Approach (short)

1. **Split:** time-based holdout — train through 2025-09-30, validate on October 2025 (mimics the real Nov–Dec shift).
2. **Cleaning:** median-impute missing `weight` / `market_index`; clip distances for safe transforms.
3. **Features:** distance/weight, geo (haversine, lat/lon deltas), equipment, route keys, market × distance interactions, cyclical calendar encodings.
4. **Model:** LightGBM regressor predicting `log1p(rate_per_mile)`, then multiply by distance.
5. **December chart:** attach city coordinates from history and daily `market_index` / `quote_signal` averages from the validation period (features only, no labels).

## Submit checklist

- [ ] GitHub repo with code + this README
- [ ] `validation_predictions.csv`
- [ ] Report PDF (`reports/assessment_report.pdf`) including the December chart
- [ ] 2–3 minute Loom walkthrough
