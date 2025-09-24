"""Configuration module for setting up paths and logging."""

import os
import sys
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from dotenv import load_dotenv
from loguru import logger

warnings.filterwarnings(
    "ignore",
    message="dropout2d: Received a 3D input to dropout2d and assuming that channel-wise 1D",
)

# --- Basic Loguru Configuration ---
# Remove any existing default handlers. This ensures a clean slate.
logger.remove()

# Add a basic, console-only logger as the default.
# This will be used if setup_logging is not called.
# Level "INFO" is a good default.
logger.add(sys.stderr, level="DEBUG")

# If tqdm is installed, configure loguru with tqdm.write for the console.
try:
    from tqdm import tqdm

    # We replace the default stderr sink with the tqdm-aware one.
    logger.remove(0)  # Remove the sink we just added at index 0
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="DEBUG")
except Exception:  # noqa: S110
    pass

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset directories
TUH_DATA_DIR = DATA_DIR / "TUABv3.0.1"

# TZINFO
TZINFO = ZoneInfo("Europe/Berlin")  # Set your timezone here

# Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set torch device
DEVICE = torch.device("cuda" if torch.cuda.device_count() >= 1 else "cpu")
logger.info(f"Torch device: {DEVICE}, Total CUDA devices: {torch.cuda.device_count()}")
