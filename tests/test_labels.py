import pandas as pd
import pytest

from ta.data.index import build_index, cross_check_filenames, load_metadata
from ta.data.labels import S3_VOWEL_DEPENDENT, TASKS, get_label, get_labels, load_letter_labels


@pytest.fixture(scope="module")
def meta():
    return load_metadata()


@pytest.fixture(scope="module")
def letters():
    return load_letter_labels()


def test_get_label_reads_metadata_column(meta):
    row = meta.iloc[0]
    for t in TASKS:
        assert get_label(t, row) == int(row[f"y_{t}"])


def test_get_label_rejects_unknown_task(meta):
    with pytest.raises(ValueError):
        get_label("S2", meta.iloc[0])


def test_labels_binary(meta):
    for t in TASKS:
        assert set(get_labels(t, meta)) == {0, 1}


def _benar_plain(meta, letters):
    ben = meta[(meta.category == "benar") & meta["item"].str.fullmatch(r"\d+")]
    return ben.merge(letters, left_on="huruf_no", right_on="no_huruf")


@pytest.mark.parametrize("task", ["S1", "T2"])
def test_letter_static_labels_match_csv(meta, letters, task):
    m = _benar_plain(meta, letters)
    assert (m[f"y_{task}"] == m[task]).all()


def test_s3_matches_csv_except_vowel_dependent(meta, letters):
    m = _benar_plain(meta, letters)
    mism = set(m.loc[m.y_S3 != m.S3, "huruf_no"])
    assert mism == set(S3_VOWEL_DEPENDENT)
    # huruf 7, 10, 19: tafkhim (1) kecuali kasrah (sub=2)
    for h in S3_VOWEL_DEPENDENT:
        sub = m[m.huruf_no == h]
        assert (sub.loc[sub["sub"] == 2, "y_S3"] == 0).all()
        assert (sub.loc[sub["sub"] != 2, "y_S3"] == 1).all()


def test_salah_is_inverse_of_benar(meta):
    g = meta.groupby(["category", "item", "sub"])[[f"y_{t}" for t in TASKS]]
    assert (g.nunique() == 1).all().all()
    first = g.first()
    assert ((1 - first.loc["benar"]) == first.loc["salah"].loc[first.loc["benar"].index]).all().all()


def test_letter_class_counts(letters):
    assert int(letters.S1.sum()) == 10
    assert int(letters.S3.sum()) == 7
    assert int(letters.T2.sum()) == 5


def test_variant_labels(meta):
    ben = meta[meta.category == "benar"]
    expected = {
        "25i": (0, 0, 0), "25ie": (0, 0, 0), "25ik": (0, 0, 0), "26t": (0, 0, 0),
        "2t": (0, 0, 1), "5t": (0, 0, 1), "8t": (0, 0, 1), "16t": (0, 1, 1), "21t": (0, 1, 1),
    }
    for item, exp in expected.items():
        rows = ben[ben["item"] == item]
        assert len(rows) > 0
        for t, v in zip(TASKS, exp):
            assert set(rows[f"y_{t}"]) == {v}, (item, t)


def test_metadata_shape(meta):
    assert len(meta) == 21591
    assert meta.category.value_counts().to_dict() == {"salah": 10800, "benar": 10791}


def test_index_counts_after_exclusion():
    idx = build_index()
    assert len(idx) == 21586
    assert (idx.split == "uji").sum() == 4311
    assert set(idx.is_correct) == {0, 1}
    assert (idx.is_correct == (idx.category == "benar").astype(int)).all()


def test_filenames_consistent_with_metadata(meta):
    assert cross_check_filenames(meta) == []
