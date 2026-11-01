"""Tes anti-leakage (CLAUDE.md bagian 4)."""
import numpy as np
import pandas as pd
import pytest

from ta.data.index import FORBIDDEN_FEATURE_COLUMNS, build_index
from ta.data.labels import TASKS
from ta.data.preprocess import DEFAULT_PREPROCESS, cache_dir
from ta.data.split import LIGO_LETTER_FOLDS, add_ligo_fold, cv_folds, ligo_folds, train_test_indices
from ta.features import assert_no_forbidden
from ta.features.mfcc import feature_names


@pytest.fixture(scope="module")
def idx():
    return build_index()


def test_grup_cv_in_single_split_and_fold(idx):
    g = idx.groupby("grup_cv").agg(n_split=("split", "nunique"), n_fold=("fold", "nunique"))
    assert (g.n_split == 1).all()
    assert (g.n_fold == 1).all()


def test_test_split_is_fold0_only(idx):
    assert set(idx.loc[idx.split == "uji", "fold"]) == {0}
    assert 0 not in set(idx.loc[idx.split == "latih", "fold"])


def test_md5_not_shared_between_train_and_test(idx):
    lat = set(idx.loc[idx.split == "latih", "md5"])
    uji = set(idx.loc[idx.split == "uji", "md5"])
    assert not (lat & uji)


def test_file_hash_not_shared_between_train_and_test(idx):
    stats = cache_dir(DEFAULT_PREPROCESS) / "stats.csv"
    if not stats.exists():
        pytest.skip("cache praproses belum dibangun (python -m ta.data.audit)")
    s = idx.merge(pd.read_csv(stats, usecols=["source_id", "md5_file", "md5_samples"]), on="source_id")
    assert len(s) == len(idx)
    assert (s.md5_samples == s.md5).all()
    lat = set(s.loc[s.split == "latih", "md5_file"])
    uji = set(s.loc[s.split == "uji", "md5_file"])
    assert not (lat & uji)


def test_cv_folds_disjoint_groups(idx):
    for f, tr, va in cv_folds(idx):
        assert not set(idx.loc[tr, "grup_cv"]) & set(idx.loc[va, "grup_cv"])
        assert (idx.loc[np.concatenate([tr, va]), "split"] == "latih").all()
    tr, te = train_test_indices(idx)
    assert not set(idx.loc[tr, "grup_cv"]) & set(idx.loc[te, "grup_cv"])


def test_ligo_partition(idx):
    letters = [h for hs in LIGO_LETTER_FOLDS.values() for h in hs]
    assert sorted(letters) == list(range(1, 29))
    li = add_ligo_fold(idx)
    # varian ikut huruf dasarnya: satu huruf hanya di satu lipatan
    assert (li.groupby("huruf_no").ligo_fold.nunique() == 1).all()
    # qalqalah tidak menumpuk di satu lipatan
    q = li[(li.category == "benar") & (li.y_T2 == 1)].groupby("ligo_fold").huruf_no.nunique()
    assert len(q) >= 5 and q.max() == 1


def test_ligo_train_test_disjoint(idx):
    for k, tr, te in ligo_folds(idx):
        assert not set(idx.loc[tr, "item"]) & set(idx.loc[te, "item"])
        assert not set(idx.loc[tr, "grup_cv"]) & set(idx.loc[te, "grup_cv"])
        for t in TASKS:
            assert set(idx.loc[te, f"y_{t}"]) == {0, 1}


def test_is_correct_not_a_feature():
    names = feature_names()
    assert "is_correct" not in names
    assert not set(names) & set(FORBIDDEN_FEATURE_COLUMNS)
    with pytest.raises(ValueError):
        assert_no_forbidden(names + ["is_correct"])


def test_feature_extraction_uses_audio_only():
    import inspect
    from ta.features.mfcc import mfcc_summary
    assert list(inspect.signature(mfcc_summary).parameters) == ["y", "sr", "p"]
