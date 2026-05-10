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

train = pd.read_csv("../data/train.csv")
val = pd.read_csv("../data/val.csv")

train.head()

train.shape

train.info()

train.describe()

train.isna().sum()

train.trip_duration.describe()

train.trip_duration.hist(bins=50)

train.passenger_count.value_counts()

train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])

train["log_trip_duration"] = np.log1p(train["trip_duration"])

train = train[train.trip_duration < 10000]

train = train[train.passenger_count > 0]
train = train[train.passenger_count <= 6]

train = train[train.trip_duration > 10]

train = train[
    (train.pickup_latitude.between(40, 42)) &
    (train.pickup_longitude.between(-75, -72)) &
    (train.dropoff_latitude.between(40, 42)) &
    (train.dropoff_longitude.between(-75, -72))]

train.trip_duration.describe()

for df in [train, val]:
    df["log_trip_duration"] = np.log1p(df["trip_duration"])

train.trip_duration.hist(bins=50)

train["log_trip_duration"].hist(bins=50)

# Feature Engineering:

train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
val["pickup_datetime"] = pd.to_datetime(val["pickup_datetime"])

for df in [train, val]:
    df["hour"] = df.pickup_datetime.dt.hour
    df["dayofweek"] = df.pickup_datetime.dt.dayofweek
    df["month"] = df.pickup_datetime.dt.month
    df["dayofyear"] = df.pickup_datetime.dt.dayofyear

for df in [train, val]:
    df["direction"] = np.arctan2(
        df["dropoff_latitude"] - df["pickup_latitude"],
        df["dropoff_longitude"] - df["pickup_longitude"]
    )

for df in [train, val]:
    df["rush_hour"] = df["hour"].apply(lambda x: 1 if (7 <= x <= 9 or 16 <= x <= 19) else 0)

for df in [train, val]:
    df["weekend"] = df["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


for df in [train, val]:
    df["distance"] = haversine(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"]
    )

for df in [train, val]:
    df["log_distance"] = np.log1p(df["distance"])

for df in [train, val]:
    df["dist_x_rush"] = df["distance"] * df["rush_hour"]
    df["distance_sq"] = df["distance"] ** 2
    df["logdist_x_hour"] = df["log_distance"] * df["hour"]

for df in [train, val]:
    df["manhattan_distance"] = (
            abs(df["pickup_latitude"] - df["dropoff_latitude"]) +
            abs(df["pickup_longitude"] - df["dropoff_longitude"])
    )

for df in [train, val]:
    df["lat_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["lon_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]

for df in [train, val]:
    df["passenger_count"] = df["passenger_count"].clip(0, 6)

for df in [train, val]:
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"Y": 1, "N": 0})

categorical_features = ["vendor_id", "hour", "dayofweek",
                        "month", "rush_hour", "weekend"]

numeric_features = ["passenger_count", "store_and_fwd_flag",
                    "distance", "log_distance", "manhattan_distance",
                    "lat_diff", "lon_diff", "direction",
                    "dist_x_rush", "distance_sq", "logdist_x_hour"]

train_features = categorical_features + numeric_features

X_train = train[train_features]
y_train = train["log_trip_duration"]

X_val = val[train_features]
y_val = val["log_trip_duration"]

print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)

# ── Pipeline ──────────────────────────────────────
column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ('scaling', StandardScaler(), numeric_features)])

pipeline = Pipeline(steps=[
    ('transform', column_transformer),
    ('regression', Ridge(alpha=1))])

# ── Train ──────────────────────────────────────────
pipeline.fit(X_train[train_features], y_train)

# ── Evaluate ───────────────────────────────────────
y_pred = pipeline.predict(X_val[train_features])

r2 = r2_score(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"R2:   {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# ── Save Model ─────────────────────────────────────
joblib.dump(pipeline, "../models/model.pkl")
print("Model saved → model.pkl")
