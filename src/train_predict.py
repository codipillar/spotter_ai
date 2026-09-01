"""Train a freight-rate model and write assessment deliverables."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, add_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])


def load_validation() -> pd.DataFrame:
    return pd.read_csv(DATA / "validation.csv", parse_dates=["date"])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "r2": float(r2_score(y_true, y_pred)),
    }


def prepare_xy(
    frame: pd.DataFrame,
    feature_medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    feat = add_features(frame)
    medians = feature_medians or {
        col: float(feat[col].median(skipna=True)) for col in NUMERIC_FEATURES
    }
    for col in NUMERIC_FEATURES:
        feat[col] = feat[col].fillna(medians[col])
    for col in CATEGORICAL_FEATURES:
        feat[col] = feat[col].astype("category")
    return feat, medians


def make_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=96,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_samples=30,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
    )


def time_split(train: pd.DataFrame, holdout_days: int = 31) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = train["date"].max() - pd.Timedelta(days=holdout_days)
    return train[train["date"] <= cutoff].copy(), train[train["date"] > cutoff].copy()


def predict_rates(model: lgb.LGBMRegressor, feat: pd.DataFrame) -> np.ndarray:
    """Model predicts log(rate-per-mile); convert back to total rate."""
    log_rpm = model.predict(feat[FEATURE_COLUMNS])
    rpm = np.expm1(log_rpm)
    rpm = np.clip(rpm, 0.05, None)
    distance = feat["distance"].to_numpy(dtype=float)
    return np.clip(rpm * np.clip(distance, 1.0, None), 1.0, None)


def train_and_evaluate(train: pd.DataFrame) -> tuple[lgb.LGBMRegressor, dict, dict[str, float]]:
    tr, va = time_split(train)
    tr_f, medians = prepare_xy(tr)
    va_f, _ = prepare_xy(va, medians)

    y_tr = tr["posted_rate"].to_numpy(dtype=float)
    y_va = va["posted_rate"].to_numpy(dtype=float)
    rpm_tr = y_tr / np.clip(tr["distance"].to_numpy(dtype=float), 1.0, None)

    model = make_model()
    model.fit(
        tr_f[FEATURE_COLUMNS],
        np.log1p(rpm_tr),
        eval_X=va_f[FEATURE_COLUMNS],
        eval_y=np.log1p(y_va / np.clip(va["distance"].to_numpy(dtype=float), 1.0, None)),
        eval_metric="l1",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    pred_va = predict_rates(model, va_f)
    score = metrics(y_va, pred_va)

    rpm = float(np.median(rpm_tr))
    baseline = np.clip(va["distance"].to_numpy(dtype=float) * rpm, 1.0, None)
    baseline_score = metrics(y_va, baseline)

    report = {
        "split": {
            "train_rows": int(len(tr)),
            "valid_rows": int(len(va)),
            "train_end": str(tr["date"].max().date()),
            "valid_start": str(va["date"].min().date()),
            "valid_end": str(va["date"].max().date()),
        },
        "model": "LightGBMRegressor predicting log1p(rate_per_mile)",
        "holdout_metrics": score,
        "distance_rpm_baseline_metrics": baseline_score,
        "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators) or model.n_estimators),
        "feature_importance_top": sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:15],
    }
    return model, report, medians


def refit_full(train: pd.DataFrame, best_iteration: int) -> tuple[lgb.LGBMRegressor, dict[str, float]]:
    feat, medians = prepare_xy(train)
    y = train["posted_rate"].to_numpy(dtype=float)
    rpm = y / np.clip(train["distance"].to_numpy(dtype=float), 1.0, None)
    model = make_model()
    model.set_params(n_estimators=max(150, best_iteration))
    model.fit(
        feat[FEATURE_COLUMNS],
        np.log1p(rpm),
        categorical_feature=CATEGORICAL_FEATURES,
    )
    return model, medians


def predict_frame(model: lgb.LGBMRegressor, frame: pd.DataFrame, medians: dict[str, float]) -> np.ndarray:
    feat, _ = prepare_xy(frame, medians)
    return predict_rates(model, feat)


def city_coords(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    for _, row in train[["pickup", "pickup_lat", "pickup_lon"]].drop_duplicates("pickup").iterrows():
        coords[str(row["pickup"])] = (float(row["pickup_lat"]), float(row["pickup_lon"]))
    for _, row in train[["delivery", "delivery_lat", "delivery_lon"]].drop_duplicates("delivery").iterrows():
        coords.setdefault(str(row["delivery"]), (float(row["delivery_lat"]), float(row["delivery_lon"])))
    return coords


def build_december_frame(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Complete december_chart_inputs with coords + daily market signals."""
    template = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    coords = city_coords(train)

    daily = (
        validation.groupby(validation["date"].dt.normalize())[["market_index", "quote_signal"]]
        .mean()
        .rename_axis("date")
        .reset_index()
    )

    out = template.drop(columns=["predicted_rate"], errors="ignore").copy()
    out["date"] = pd.to_datetime(out["date"])
    pickup_xy = out["pickup"].map(coords)
    delivery_xy = out["delivery"].map(coords)
    out["pickup_lat"] = pickup_xy.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    out["pickup_lon"] = pickup_xy.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)
    out["delivery_lat"] = delivery_xy.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    out["delivery_lon"] = delivery_xy.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)

    out = out.merge(daily, on="date", how="left")
    out["market_index"] = out["market_index"].fillna(validation["market_index"].median())
    out["quote_signal"] = out["quote_signal"].fillna(validation["quote_signal"].median())
    out["load_id"] = [f"DEC-{i:02d}" for i in range(1, len(out) + 1)]
    return out


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train = load_train()
    validation = load_validation()

    print("Evaluating with time-based holdout (last ~31 days of train)...")
    _, report, _ = train_and_evaluate(train)
    print(json.dumps(report["holdout_metrics"], indent=2))
    print("Baseline:", json.dumps(report["distance_rpm_baseline_metrics"], indent=2))

    print("Refitting on full train_test.csv...")
    model, medians = refit_full(train, report["best_iteration"])

    val_pred = predict_frame(model, validation, medians)
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    pred_df = template.copy()
    pred_map = dict(zip(validation["load_id"].astype(str), val_pred))
    pred_df["predicted_rate"] = pred_df["load_id"].astype(str).map(pred_map)
    if pred_df["predicted_rate"].isna().any():
        missing = int(pred_df["predicted_rate"].isna().sum())
        raise SystemExit(f"Missing predictions for {missing} load_ids")
    out_pred = ROOT / "validation_predictions.csv"
    pred_df.to_csv(out_pred, index=False)
    print(f"Wrote {out_pred} ({len(pred_df):,} rows)")

    december_frame = build_december_frame(train, validation)
    december_rates = predict_frame(model, december_frame, medians)
    december_out = pd.read_csv(DATA / "december_chart_inputs.csv")
    december_out["predicted_rate"] = december_rates
    december_path = DATA / "december_chart_inputs.csv"
    december_out.to_csv(december_path, index=False)
    december_out.to_csv(ROOT / "december-chart-inputs.csv", index=False)
    print(
        f"Wrote {december_path} "
        f"(Dec rate std={float(np.std(december_rates)):.2f}, "
        f"range={float(np.min(december_rates)):.1f}-{float(np.max(december_rates)):.1f})"
    )

    with open(ARTIFACTS / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(ARTIFACTS / "model.pkl", "wb") as fh:
        pickle.dump({"model": model, "medians": medians, "features": FEATURE_COLUMNS}, fh)

    importance = pd.DataFrame(
        report["feature_importance_top"], columns=["feature", "importance"]
    )
    importance.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    print("Saved artifacts to", ARTIFACTS)


if __name__ == "__main__":
    main()
