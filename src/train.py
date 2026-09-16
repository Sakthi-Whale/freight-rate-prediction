from joblib import dump

from scipy.stats import randint, uniform

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)

from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from src.config import MODEL_PATH
from src.data_loader import load_datasets
from src.preprocessing import (
    prepare_features,
    build_preprocessor,
)
from src.evaluate import evaluate_regression

def train_pipeline():

    datasets = load_datasets()

    X, y = prepare_features(
        datasets["train"]
    )

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        ))
    ])

    params = {
        "model__n_estimators": randint(300, 1000),
        "model__max_depth": randint(4, 12),
        "model__learning_rate": uniform(0.02, 0.08),
        "model__min_child_weight": randint(1, 8),
        "model__subsample": uniform(0.70, 0.30),
        "model__colsample_bytree": uniform(0.70, 0.30),
        "model__reg_alpha": uniform(0, 1),
        "model__reg_lambda": uniform(1, 3),
    }

    search = RandomizedSearchCV(
        pipeline,
        params,
        n_iter=12,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=2,
        verbose=2,
    )

    search.fit(X_train, y_train)

    model = search.best_estimator_

    predictions = model.predict(X_valid)

    metrics = evaluate_regression(
        y_valid,
        predictions,
    )

    print(
        f"Best CV RMSE : {-search.best_score_:.2f}"
    )

    print(
        f"RMSE : {metrics['rmse']:.2f}"
    )
    print(
        f"MAE  : {metrics['mae']:.2f}"
    )
    print(
        f"R²   : {metrics['r2']:.4f}"
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)

    dump(model, MODEL_PATH)

    return model