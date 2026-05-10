import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# ── Load Model ─────────────────────────────────────
pipeline = joblib.load("../models/model.pkl")
print("Model loaded ✓")

# ── Load Test CSV ──────────────────────────────────
test = pd.read_csv("../data/test.csv")


# ── Feature Engineering ────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def add_features(df):
    df = df.copy()
    df["pickup_datetime"]    = pd.to_datetime(df["pickup_datetime"])
    df["hour"]               = df["pickup_datetime"].dt.hour
    df["dayofweek"]          = df["pickup_datetime"].dt.dayofweek
    df["month"]              = df["pickup_datetime"].dt.month
    df["dayofyear"]          = df["pickup_datetime"].dt.dayofyear
    df["rush_hour"]          = df["hour"].apply(lambda x: 1 if (7<=x<=9 or 16<=x<=19) else 0)
    df["weekend"]            = df["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    df["distance"]           = haversine(df["pickup_latitude"], df["pickup_longitude"],
                                          df["dropoff_latitude"], df["dropoff_longitude"])
    df["log_distance"]       = np.log1p(df["distance"])
    df["manhattan_distance"] = (abs(df["pickup_latitude"]  - df["dropoff_latitude"]) +
                                 abs(df["pickup_longitude"] - df["dropoff_longitude"]))
    df["lat_diff"]           = df["dropoff_latitude"]  - df["pickup_latitude"]
    df["lon_diff"]           = df["dropoff_longitude"] - df["pickup_longitude"]
    df["direction"]          = np.arctan2(df["lat_diff"], df["lon_diff"])
    df["dist_x_rush"]        = df["distance"] * df["rush_hour"]
    df["distance_sq"]        = df["distance"] ** 2
    df["logdist_x_hour"]     = df["log_distance"] * df["hour"]
    df["passenger_count"]    = df["passenger_count"].clip(0, 6)
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"Y": 1, "N": 0})
    return df

# ── Prepare Test ───────────────────────────────────
test = add_features(test)

categorical_features = ["vendor_id", "hour", "dayofweek", "month", "rush_hour", "weekend"]
numeric_features     = ["passenger_count", "store_and_fwd_flag", "distance", "log_distance",
                         "manhattan_distance", "lat_diff", "lon_diff", "direction",
                         "dist_x_rush", "distance_sq", "logdist_x_hour"]

X_test = test[categorical_features + numeric_features]
y_test = np.log1p(test["trip_duration"])

# ── Predict & Score ────────────────────────────────
y_pred = pipeline.predict(X_test)

print(f"Test R²:   {r2_score(y_test, y_pred):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")