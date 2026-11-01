"""Path proyek. Semua path relatif ke root repo (folder yang berisi pyproject.toml)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABELS_DIR = DATA_DIR / "labels"
INTERIM_DIR = DATA_DIR / "interim"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = ROOT / "reports"
RUNS_DIR = ROOT / "runs"
CONFIGS_DIR = ROOT / "configs"

METADATA_CSV = LABELS_DIR / "metadata_rekaman.csv"
LETTER_LABELS_CSV = LABELS_DIR / "label_per_huruf.csv"
DATASET_EXCLUDED_CSV = LABELS_DIR / "rekaman_dikeluarkan.csv"
LEAKAGE_EXCLUDED_CSV = LABELS_DIR / "exclude_leakage.csv"

CATEGORY_DIRS = {"benar": RAW_DIR / "Data Benar", "salah": RAW_DIR / "Data Salah"}
