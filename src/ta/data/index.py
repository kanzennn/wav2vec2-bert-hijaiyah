"""Indeks rekaman: metadata + path berkas + is_correct, dikurangi rekaman yang dikeluarkan."""
import re

import pandas as pd

from ta.paths import CATEGORY_DIRS, DATASET_EXCLUDED_CSV, LEAKAGE_EXCLUDED_CSV, METADATA_CSV

# nama berkas: <item>-<harakat>-<pengulangan>.wav, plus akhiran "-error" pada 26 berkas di Data Benar/15
FILENAME_RE = re.compile(r"^(\d+)([A-Za-z]*)-(\d)-(\d+)(-error)?\.wav$")

# kolom metadata yang tidak boleh jadi fitur model (identitas, label, dan status benar/salah)
FORBIDDEN_FEATURE_COLUMNS = (
    "is_correct", "category", "folder", "filename", "source_id", "responden", "grup_cv",
    "item", "huruf_no", "sub", "rep", "flag_error", "md5", "split", "fold", "kode_uji",
    "y_S1", "y_S3", "y_T2",
)


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(METADATA_CSV, dtype={"item": str, "kode_uji": str})


def load_leakage_exclusions() -> pd.DataFrame:
    if not LEAKAGE_EXCLUDED_CSV.exists():
        return pd.DataFrame(columns=["source_id"])
    return pd.read_csv(LEAKAGE_EXCLUDED_CSV)


def recording_path(category: str, folder, filename: str):
    return CATEGORY_DIRS[category] / str(folder) / filename


def parse_filename(filename: str) -> dict:
    m = FILENAME_RE.match(filename)
    if m is None:
        raise ValueError(f"nama berkas tidak standar: {filename}")
    return {"huruf_no": int(m.group(1)), "varian": m.group(2), "harakat": int(m.group(3)),
            "pengulangan": int(m.group(4)), "is_error_name": m.group(5) is not None}


def build_index(apply_exclusions: bool = True) -> pd.DataFrame:
    """Satu baris per rekaman terpakai, urutan stabil (urutan metadata)."""
    df = load_metadata()
    df["is_correct"] = (df["category"] == "benar").astype(int)
    df["path"] = [str(recording_path(c, f, n)) for c, f, n in zip(df["category"], df["folder"], df["filename"])]
    if apply_exclusions:
        excl = set(load_leakage_exclusions()["source_id"])
        df = df[~df["source_id"].isin(excl)]
    return df.reset_index(drop=True)


def keep_without_flag_error(df: pd.DataFrame) -> pd.Series:
    """Mask baris yang dipertahankan bila 26 berkas `-error` (flag_error) dikeluarkan dari data latih.

    Untuk analisis sensitivitas E2 saja (CLAUDE.md 10.5). Data uji tidak pernah disentuh.
    """
    return ~(df["flag_error"].astype(bool) & (df["split"] == "latih"))


def cross_check_filenames(df: pd.DataFrame) -> list:
    """Bandingkan hasil parse nama berkas dengan kolom metadata. Mengembalikan daftar ketidakcocokan."""
    bad = []
    for r in df.itertuples():
        p = parse_filename(r.filename)
        if (p["huruf_no"], p["harakat"], p["pengulangan"]) != (r.huruf_no, r.sub, r.rep) \
                or f"{p['huruf_no']}{p['varian']}" != r.item or p["is_error_name"] != bool(r.flag_error):
            bad.append(r.source_id)
    return bad


def disk_vs_metadata() -> dict:
    """Berkas WAV di disk vs metadata (sebelum pengecualian kebocoran)."""
    meta_ids = set(load_metadata()["source_id"])
    disk_ids = set()
    for cat, d in CATEGORY_DIRS.items():
        for p in d.rglob("*.wav"):
            disk_ids.add(f"{cat}/{p.parent.name}/{p.name}")
    excl = pd.read_csv(DATASET_EXCLUDED_CSV)
    excl_ids = {f"{r.category}/{r.folder}/{r.filename}" for r in excl.itertuples()}
    return {
        "n_disk": len(disk_ids),
        "n_disk_per_category": {c: sum(i.startswith(c + "/") for i in disk_ids) for c in CATEGORY_DIRS},
        "missing_on_disk": sorted(meta_ids - disk_ids),
        "not_in_metadata": sorted(disk_ids - meta_ids),
        "not_in_metadata_equals_dataset_excluded": (disk_ids - meta_ids) == excl_ids,
    }
