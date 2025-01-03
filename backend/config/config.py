import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# App settings
APP_NAME = "Media Mix Modeling"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Model settings
AVAILABLE_MODELS = {
    "BMMM": "Bayesian Media Mix Model",
    "LightGBM": "LightGBM Regressor",
    "XGBoost": "XGBoost Regressor",
    "Meta_Robyn": "Meta Robyn"
}

# Data settings
ALLOWED_EXTENSIONS = [".csv", ".xlsx"]
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

# Model parameters
DEFAULT_TRAIN_TEST_SPLIT = 0.2
DEFAULT_RANDOM_STATE = 42 