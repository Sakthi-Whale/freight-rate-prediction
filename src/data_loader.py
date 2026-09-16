import pandas as pd
from src.config import (
    TRAIN_FILE,
    VALIDATION_FILE,
    DECEMBER_FILE,
)

def load_datasets():
    return {
        "train": pd.read_csv(TRAIN_FILE),
        "validation": pd.read_csv(VALIDATION_FILE),
        "december": pd.read_csv(DECEMBER_FILE),
    }