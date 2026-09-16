import pandas as pd

from joblib import load

from src.config import (
    MODEL_PATH,
    TRAIN_FILE,
    VALIDATION_FILE,
    TEMPLATE_FILE,
    DECEMBER_FILE,
)

from src.preprocessing import prepare_features

def build_lookup(train_df):

    pickup = (
        train_df
        .groupby("pickup")
        [["pickup_lat","pickup_lon"]]
        .median()
    )

    delivery = (
        train_df
        .groupby("delivery")
        [["delivery_lat","delivery_lon"]]
        .median()
    )

    return pickup, delivery

def generate_predictions():

    model = load(MODEL_PATH)

    train = pd.read_csv(TRAIN_FILE)

    pickup_lookup, delivery_lookup = (
        build_lookup(train)
    )

    validation = pd.read_csv(VALIDATION_FILE)
    template = pd.read_csv(TEMPLATE_FILE)

    val_features = prepare_features(
        validation,
        training=False,
    )

    template["predicted_rate"] = (
        model.predict(val_features)
    )

    template.to_csv(
        "validation_predictions.csv",
        index=False,
    )

    december = pd.read_csv(DECEMBER_FILE)

    december = december.join(
        pickup_lookup,
        on="pickup",
    )

    december = december.join(
        delivery_lookup,
        on="delivery",
    )

    december["market_index"] = (
        train["market_index"].median()
    )

    december["quote_signal"] = (
        train["quote_signal"].median()
    )

    dec_features = prepare_features(
        december,
        training=False,
    )

    december["predicted_rate"] = (
        model.predict(dec_features)
    )

    december.to_csv(
        "december_predictions.csv",
        index=False,
    )