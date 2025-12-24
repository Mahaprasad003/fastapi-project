import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
DATA_FILE_NAME = 'car-details.csv'
DATA_FILE_PATH = DATA_DIR / DATA_FILE_NAME

MODEL_DIR = BASE_DIR / 'app' / 'models'
MODEL_NAME = 'model.joblib'
MODEL_PATH = MODEL_DIR / MODEL_NAME
