import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.feature_engineering import create_features

def prepare_features(df, training=True):
    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek

    data["weight"] = data["weight"].abs()

    data["market_index"] = (
        data["market_index"]
        .fillna(data["market_index"].median())
    )

    data = create_features(data)

    data.drop(columns=["date"], inplace=True)

    if training:
        y = data.pop("posted_rate")
        data.drop(columns=["load_id"], inplace=True)
        return data, y

    if "load_id" in data.columns:
        data.drop(columns=["load_id"], inplace=True)

    return data

def build_preprocessor():

    categorical = [
        "pickup",
        "delivery",
        "equipment",
        "pickup_state",
        "delivery_state",
    ]

    numerical = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "geo_distance",
        "weight",
        "weight_per_mile",
        "market_index",
        "market_distance",
        "quote_signal",
        "year",
        "month",
        "day",
        "day_of_week",
    ]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numerical),
        ("cat", categorical_pipe, categorical)
    ])