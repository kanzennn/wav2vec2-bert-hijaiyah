"""Label sifat huruf. Sumber satu-satunya: kolom y_* di metadata_rekaman.csv.

Label Data Salah sudah dibalik oleh dataset, jadi di sini TIDAK dibalik lagi.
"""
import pandas as pd

from ta.paths import LETTER_LABELS_CSV

TASKS = ("S1", "S3", "T2")
TASK_NAMES = {
    "S1": ("jahr", "hams"),
    "S3": ("istifal", "isti'la"),
    "T2": ("tanpa qalqalah", "qalqalah"),
}
# huruf yang S3-nya bergantung harakat (tafkhim kecuali kasrah), jadi beda dari label per huruf
S3_VOWEL_DEPENDENT = (7, 10, 19)


def label_column(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"task tidak dikenal: {task!r}, harus salah satu dari {TASKS}")
    return f"y_{task}"


def get_label(task: str, row) -> int:
    """Label biner (0/1) satu rekaman untuk sifat `task`; `row` = baris metadata (Series/dict)."""
    value = int(row[label_column(task)])
    if value not in (0, 1):
        raise ValueError(f"label {task} tidak biner: {value}")
    return value


def get_labels(task: str, df: pd.DataFrame):
    """Versi vektor dari get_label untuk satu DataFrame."""
    y = df[label_column(task)].astype(int).to_numpy()
    if not set(y.tolist()) <= {0, 1}:
        raise ValueError(f"label {task} tidak biner")
    return y


def load_letter_labels() -> pd.DataFrame:
    return pd.read_csv(LETTER_LABELS_CSV, encoding="utf-8-sig")
