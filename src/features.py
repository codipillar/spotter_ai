"""Feature engineering for freight rate prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT_ORDER = ["Dry Van", "Reefer", "Flatbed"]


def haversine_miles(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in miles."""
    r = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with model features added."""
    out = df.copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out["distance"] = pd.to_numeric(out["distance"], errors="coerce")
    out["market_index"] = pd.to_numeric(out["market_index"], errors="coerce")
    out["quote_signal"] = pd.to_numeric(out["quote_signal"], errors="coerce")
    for col in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Calendar
    out["dayofweek"] = out["date"].dt.dayofweek.astype("int16")
    out["month"] = out["date"].dt.month.astype("int16")
    out["day"] = out["date"].dt.day.astype("int16")
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype("int16")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")
    out["dayofyear"] = out["date"].dt.dayofyear.astype("int16")
    # Cyclical calendar encodings help extrapolate past the training window.
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)

    # Geography / route
    out["abs_lat_diff"] = (out["pickup_lat"] - out["delivery_lat"]).abs()
    out["abs_lon_diff"] = (out["pickup_lon"] - out["delivery_lon"]).abs()
    out["haversine_miles"] = haversine_miles(
        out["pickup_lat"].to_numpy(),
        out["pickup_lon"].to_numpy(),
        out["delivery_lat"].to_numpy(),
        out["delivery_lon"].to_numpy(),
    )
    out["distance_gap"] = out["distance"] - out["haversine_miles"]
    out["route_key"] = out["pickup"].astype(str) + "->" + out["delivery"].astype(str)

    # Load economics proxies
    out["weight_per_mile"] = out["weight"] / out["distance"].clip(lower=1.0)
    out["log_distance"] = np.log1p(out["distance"].clip(lower=0.0))
    out["log_weight"] = np.log1p(out["weight"].clip(lower=0.0))
    out["market_x_quote"] = out["market_index"] * out["quote_signal"]
    out["market_x_distance"] = out["market_index"] * out["log_distance"]
    out["quote_x_distance"] = out["quote_signal"] * out["log_distance"]

    # Equipment as ordered categorical codes (stable across train/infer)
    out["equipment"] = pd.Categorical(out["equipment"], categories=EQUIPMENT_ORDER)
    out["equipment_code"] = out["equipment"].cat.codes.astype("int16")

    return out


NUMERIC_FEATURES = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "dayofweek",
    "month",
    "day",
    "weekofyear",
    "is_weekend",
    "dayofyear",
    "doy_sin",
    "doy_cos",
    "dow_sin",
    "dow_cos",
    "abs_lat_diff",
    "abs_lon_diff",
    "haversine_miles",
    "distance_gap",
    "weight_per_mile",
    "log_distance",
    "log_weight",
    "market_x_quote",
    "market_x_distance",
    "quote_x_distance",
    "equipment_code",
]

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route_key",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
