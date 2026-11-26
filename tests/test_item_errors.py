"""Tes E5: galat per item dihitung per huruf induk (huruf_no), varian digabung."""
import numpy as np
import pandas as pd

from ta.analysis import item_errors


def test_item_errors_merges_variants_into_parent_letter():
    rows = []
    for resp in ("benar-1", "benar-2", "salah-1", "salah-2"):
        for item, h, wrong in (("2", 2, 1), ("2t", 2, 0), ("25i", 25, 1), ("25ie", 25, 1), ("25ik", 25, 0), ("1", 1, 0)):
            rows.append({"responden": resp, "item": item, "huruf_no": h, "y_true": 1, "y_pred": 1 - wrong})
    t = item_errors("T2", "dummy", pd.DataFrame(rows), np.random.default_rng(0))
    assert sorted(t.huruf_no) == [1, 2, 25]
    t = t.set_index("huruf_no")
    assert t.loc[2, "item_tergabung"] == "2+2t"
    assert t.loc[25, "item_tergabung"] == "25i+25ie+25ik"
    assert t.loc[2, "n"] == 8 and np.isclose(t.loc[2, "galat"], 0.5)
    assert t.loc[25, "n"] == 12 and np.isclose(t.loc[25, "galat"], 2 / 3)
    assert t.loc[1, "galat"] == 0
