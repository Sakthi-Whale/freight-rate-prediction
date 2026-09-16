from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

def evaluate_regression(y_true, y_pred):

    return {
        "rmse": root_mean_squared_error(
            y_true,
            y_pred
        ),
        "mae": mean_absolute_error(
            y_true,
            y_pred
        ),
        "r2": r2_score(
            y_true,
            y_pred
        ),
    }