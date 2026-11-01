"""Pemuat fitur. Fitur hanya dihitung dari audio; kolom metadata tidak pernah masuk ke X."""
from ta.data.index import FORBIDDEN_FEATURE_COLUMNS


def assert_no_forbidden(feature_names):
    bad = sorted(set(feature_names) & set(FORBIDDEN_FEATURE_COLUMNS))
    if bad:
        raise ValueError(f"kolom terlarang masuk ke fitur: {bad}")


def load_features(cfg, index):
    """Mengembalikan (X, feature_names) dengan urutan baris = urutan index."""
    kind = cfg["features"]["type"]
    if kind == "mfcc":
        from ta.features.mfcc import load_mfcc_features
        X, names = load_mfcc_features(cfg, index)
    else:
        raise ValueError(f"fitur belum didukung: {kind}")
    assert_no_forbidden(names)
    return X, names
