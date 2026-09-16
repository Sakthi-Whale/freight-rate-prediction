from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

TRAIN_FILE = DATA_DIR / "train-test.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEMPLATE_FILE = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_FILE = DATA_DIR / "december-chart-inputs.csv"

MODEL_PATH = MODEL_DIR / "best_model.pkl"