"""Split latih/uji dan fold 1-5 dari metadata (dipakai apa adanya) + lipatan leave-item-group-out (LIGO).

LIGO: 36 item dikelompokkan per huruf (varian ikut huruf dasarnya, mis. 2 dan 2t, 25i/25ie/25ik) ke 6 lipatan
berisi 6 item. Lima huruf qalqalah disebar ke lima lipatan berbeda; tiap lipatan memuat huruf hams dan jahr.
Protokol evaluasi LIGO (lihat ta.evaluate): latih = responden split latih dengan item di luar lipatan k,
uji = responden split uji dengan item di lipatan k. Jadi item DAN responden sama-sama tidak pernah dilihat.
"""
import pandas as pd

from ta.data.index import build_index
from ta.paths import SPLITS_DIR

# lipatan LIGO per nomor huruf (huruf_no -> lipatan); varian mengikuti huruf dasarnya
LIGO_LETTER_FOLDS = {
    1: [2, 3, 9, 15, 23],        # ب(q) ت(h) ذ ض(i) ل
    2: [4, 5, 11, 17, 24],       # ث(h) ج(q) ز ظ(i) م
    3: [6, 7, 8, 10, 18],        # ح(h) خ(h,i*) د(q) ر(i*) ع
    4: [1, 12, 16, 19, 28],      # ا س(h) ط(q,i) غ(i*) ي
    5: [13, 14, 20, 21, 27],     # ش(h) ص(h,i) ف(h) ق(q,i) ه(h)
    6: [22, 25, 26],             # ك(h) ن (25i, 25ie, 25ik) و (26, 26t)
}
# q = qalqalah, h = hams, i = isti'la, i* = isti'la kecuali kasrah
N_LIGO_FOLDS = len(LIGO_LETTER_FOLDS)


def letter_to_ligo_fold() -> dict:
    return {h: f for f, hs in LIGO_LETTER_FOLDS.items() for h in hs}


def add_ligo_fold(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ligo_fold"] = df["huruf_no"].map(letter_to_ligo_fold())
    if df["ligo_fold"].isna().any():
        raise ValueError("ada huruf tanpa lipatan LIGO")
    df["ligo_fold"] = df["ligo_fold"].astype(int)
    return df


def cv_folds(df: pd.DataFrame):
    """Yield (fold, idx_train, idx_val) untuk validasi silang fold 1-5 di split latih."""
    lat = df[df["split"] == "latih"]
    for f in sorted(lat["fold"].unique()):
        yield int(f), lat.index[lat["fold"] != f].to_numpy(), lat.index[lat["fold"] == f].to_numpy()


def train_test_indices(df: pd.DataFrame):
    """(idx_latih, idx_uji) untuk evaluasi akhir di fold 0."""
    return df.index[df["split"] == "latih"].to_numpy(), df.index[df["split"] == "uji"].to_numpy()


def ligo_folds(df: pd.DataFrame):
    """Yield (k, idx_train, idx_test) untuk leave-item-group-out."""
    df = add_ligo_fold(df)
    for k in range(1, N_LIGO_FOLDS + 1):
        tr = df.index[(df["split"] == "latih") & (df["ligo_fold"] != k)].to_numpy()
        te = df.index[(df["split"] == "uji") & (df["ligo_fold"] == k)].to_numpy()
        yield k, tr, te


def write_splits():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    idx = add_ligo_fold(build_index())
    idx[["source_id", "grup_cv", "split", "fold"]].to_csv(SPLITS_DIR / "split_folds.csv", index=False)
    items = idx.groupby("item", sort=False).agg(huruf_no=("huruf_no", "first"), ligo_fold=("ligo_fold", "first"))
    items.reset_index().sort_values(["ligo_fold", "huruf_no", "item"]).to_csv(SPLITS_DIR / "ligo_items.csv", index=False)
    idx[["source_id", "item", "split", "ligo_fold"]].to_csv(SPLITS_DIR / "ligo_folds.csv", index=False)
    return idx


if __name__ == "__main__":
    out = write_splits()
    print(out.groupby(["ligo_fold"])["item"].unique().to_dict())
