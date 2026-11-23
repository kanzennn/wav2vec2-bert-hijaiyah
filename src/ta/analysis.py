"""Analisis lanjutan yang dijanjikan di proposal (subbab 3.5-3.7), dihitung dari berkas prediksi di runs/.

Jalankan:
    PYTHONPATH=src python -m ta.analysis \
        --xgb runs/20261102-004931_xgb_mfcc_all --w2v runs/20261109-100209_w2v_bert_all \
        [--frozen RUN_DIR_ENCODER_BEKU] [--xgb-seeds RUN_DIR_XGB_EXT] --out reports/analysis

Isi (semua hasil EKSPLORATIF, karena data uji sudah pernah dilihat; lihat subbab 3.1 proposal):
1. Metrik uji per model, per sifat, per kategori; aturan acuan "kategori saja" dan "item+harakat saja".
2. Delta macro-F1 (Wav2Vec2-BERT - XGBoost) dengan CI bootstrap klaster tingkat responden (bertingkat per kategori),
   uji permutasi tanda berpasangan per responden, koreksi Holm atas tiga sifat.
3. LIGO per kategori (hanya subset Data Benar yang ditafsirkan sebagai generalisasi sifat akustik).
4. Sensitivitas: tanpa 26 berkas `-error`; tanpa salinan identik (md5) di dalam folder responden.
5. Galat per item dengan CI bootstrap responden.
6. Pembanding encoder beku + probe linear dan XGBoost 3 seed, bila run-nya diberikan.

Unit independen = responden (8 responden uji: 4 Data Benar + 4 Data Salah). Dengan hanya 8 klaster, CI bootstrap
cenderung terlalu sempit dan nilai p terkecil uji permutasi adalah 2/256 = 0,0078 (dua sisi).
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ta.paths import ROOT

TASKS = ["S1", "S3", "T2"]
RNG_SEED = 20261121
B_MAIN = 10000
B_LIGO = 1000
B_ITEM = 2000


# ---------- metrik dari hitungan konfusi ----------

def counts(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return np.array([((y == 1) & (p == 1)).sum(), ((y == 0) & (p == 1)).sum(),
                     ((y == 1) & (p == 0)).sum(), ((y == 0) & (p == 0)).sum()], dtype=float)  # TP FP FN TN


def macro_f1_c(c):
    """macro-F1 dua kelas dari (TP, FP, FN, TN); c bisa [..., 4]."""
    tp, fp, fn, tn = (c[..., i] for i in range(4))
    def f1(a, b, d):
        den = 2 * a + b + d
        return np.where(den > 0, 2 * a / np.where(den > 0, den, 1), 0.0)
    return 0.5 * (f1(tp, fp, fn) + f1(tn, fn, fp))


def bal_acc_c(c):
    tp, fp, fn, tn = (c[..., i] for i in range(4))
    tpr = np.where(tp + fn > 0, tp / np.where(tp + fn > 0, tp + fn, 1), 0.0)
    tnr = np.where(tn + fp > 0, tn / np.where(tn + fp > 0, tn + fp, 1), 0.0)
    return 0.5 * (tpr + tnr)


def acc_c(c):
    return (c[..., 0] + c[..., 3]) / c.sum(-1)


# ---------- pemuatan data ----------

def load_preds(run, task, kind):
    return pd.read_csv(Path(run) / f"{task}_predictions_{kind}.csv")


def load_meta():
    m = pd.read_csv(ROOT / "data/labels/metadata_rekaman.csv")
    return m.set_index("source_id")


def per_resp_counts(df, resp_order):
    """Array [n_resp, 4] hitungan konfusi per responden, urutan mengikuti resp_order."""
    out = np.zeros((len(resp_order), 4))
    for i, r in enumerate(resp_order):
        d = df[df.responden == r]
        out[i] = counts(d.y_true, d.y_pred)
    return out


def model_table(task, xgb, w2v_seeds, resp_order, extra=None):
    """Hitungan per responden: dict nama_model -> list (per seed) dari array [n_resp, 4]."""
    t = {"xgb": [per_resp_counts(xgb, resp_order)],
         "w2v_bert": [per_resp_counts(w2v_seeds[s], resp_order) for s in sorted(w2v_seeds)]}
    for name, seeds_df in (extra or {}).items():
        t[name] = [per_resp_counts(seeds_df[s], resp_order) for s in sorted(seeds_df)]
    return t


def score_from_idx(model_counts, idx):
    """Rata-rata antar-seed macro-F1 untuk sampel responden idx ([B, n] indeks) -> [B]."""
    vals = [macro_f1_c(c[idx].sum(1)) for c in model_counts]
    return np.mean(vals, axis=0)


# ---------- bootstrap klaster dan uji permutasi ----------

def strata_sample(rng, resp_order, strata, B):
    """Indeks bootstrap [B, n_resp]: responden diambil ulang dengan pengembalian di dalam tiap stratum."""
    cols = []
    for s in sorted(set(strata)):
        members = np.array([i for i, x in enumerate(strata) if x == s])
        cols.append(members[rng.integers(0, len(members), size=(B, len(members)))])
    return np.concatenate(cols, axis=1)


def delta_ci(tab, ref, resp_order, strata, B, rng):
    """CI 95% persentil untuk selisih skor model - ref, dan skor masing-masing."""
    idx = strata_sample(rng, resp_order, strata, B)
    d = score_from_idx(tab["w2v_bert"], idx) - score_from_idx(tab[ref], idx)
    return np.percentile(d, [2.5, 97.5])


def sign_flip_p(diff):
    """Uji permutasi tanda eksak (dua sisi) pada rata-rata selisih per responden."""
    n = len(diff)
    obs = abs(diff.mean())
    signs = np.array(list(itertools.product([1, -1], repeat=n)))
    perm = np.abs((signs * diff).mean(1))
    return float((perm >= obs - 1e-12).mean())


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def resp_f1(model_counts):
    """macro-F1 per responden, rata-rata antar-seed: [n_resp]."""
    return np.mean([macro_f1_c(c) for c in model_counts], axis=0)


def analyze_delta(data, label, B, rng_seed=RNG_SEED, ref="xgb", model="w2v_bert"):
    """data: {task: (tab, resp_order, strata)} -> DataFrame hasil delta per sifat (+ Holm)."""
    rows = []
    for task in TASKS:
        tab, resp_order, strata = data[task]
        rng = np.random.default_rng(rng_seed)
        tot = lambda cl: np.mean([macro_f1_c(c.sum(0)) for c in cl])
        s_m, s_r = tot(tab[model]), tot(tab[ref])
        lo, hi = _ci_generic(tab, model, ref, strata, B, rng)
        dif = resp_f1(tab[model]) - resp_f1(tab[ref])
        rows.append({"variasi": label, "sifat": task, "model": model, "pembanding": ref,
                     "macroF1_model": s_m, "macroF1_pembanding": s_r, "delta": s_m - s_r,
                     "ci95_lo": lo, "ci95_hi": hi, "ci_mencakup_nol": bool(lo <= 0 <= hi),
                     "p_permutasi": sign_flip_p(dif), "n_responden": len(resp_order),
                     "n_rekaman": int(tab[model][0].sum())})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_permutasi.values)
    return out


def _ci_generic(tab, model, ref, strata, B, rng):
    idx = strata_sample(rng, None, strata, B)
    d = score_from_idx(tab[model], idx) - score_from_idx(tab[ref], idx)
    return np.percentile(d, [2.5, 97.5])


# ---------- aturan acuan ----------

def reference_rules(task, meta, test_df):
    """Aturan acuan dari data latih: 'kategori saja' dan 'item+harakat saja' (suara diabaikan)."""
    lab = f"y_{task}"
    tr = meta[meta.split == "latih"]
    cat_major = tr.groupby("category")[lab].agg(lambda s: int(s.mean() > 0.5))
    ih_major = tr.groupby(["item", "sub"])[lab].agg(lambda s: int(s.mean() > 0.5))  # seri (0,5) jatuh ke kelas 0
    p_cat = test_df.category.map(cat_major).values
    key = list(zip(test_df.item, test_df["sub"]))
    p_ih = np.array([ih_major.get(k, 0) for k in key])
    return {"kategori_saja": p_cat, "item_harakat_saja": p_ih}


# ---------- LIGO per kategori ----------

def ligo_table(task, runs):
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for name, run in runs.items():
        d = load_preds(run, task, "ligo")
        if "seed" in d.columns:
            d = d[d.seed == d.seed.min()]
        for cat in ("all", "benar", "salah"):
            sub = d if cat == "all" else d[d.category == cat]
            for fold in ["pooled"] + sorted(sub.ligo_fold.unique().tolist()):
                x = sub if fold == "pooled" else sub[sub.ligo_fold == fold]
                if len(x) == 0:
                    continue
                c = counts(x.y_true, x.y_pred)
                auc = roc_auc_score(x.y_true, x.score) if x.y_true.nunique() == 2 else np.nan
                row = {"sifat": task, "model": name, "subset": cat, "ligo_fold": fold, "n": len(x),
                       "macro_f1": float(macro_f1_c(c)), "balanced_acc": float(bal_acc_c(c)), "auc": auc,
                       "auc_lo": np.nan, "auc_hi": np.nan}
                if fold == "pooled":
                    row["auc_lo"], row["auc_hi"] = auc_ci(x, rng)
                rows.append(row)
    return rows


def auc_ci(x, rng):
    """CI bootstrap klaster (responden, bertingkat per kategori) untuk AUC pooled."""
    groups = {cat: [(g.y_true.values, g.score.values) for _, g in d.groupby("responden")]
              for cat, d in x.groupby("category")}
    vals = []
    for _ in range(B_LIGO):
        ys, ss = [], []
        for cat, gl in groups.items():
            for i in rng.integers(0, len(gl), len(gl)):
                ys.append(gl[i][0])
                ss.append(gl[i][1])
        y = np.concatenate(ys)
        if y.min() != y.max():
            vals.append(roc_auc_score(y, np.concatenate(ss)))
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


# ---------- galat per item ----------

def item_errors(task, name, df, rng):
    """Galat per item; CI bootstrap klaster responden (bertingkat per kategori)."""
    resp = sorted(df.responden.unique())
    strata = [r.split("-")[0] for r in resp]
    items = sorted(df.item.unique())
    E = np.zeros((len(resp), len(items)))
    N = np.zeros_like(E)
    for i, r in enumerate(resp):
        d = df[df.responden == r]
        for j, it in enumerate(items):
            di = d[d.item == it]
            E[i, j], N[i, j] = (di.y_true != di.y_pred).sum(), len(di)
    idx = strata_sample(rng, resp, strata, B_ITEM)
    be, bn = E[idx].sum(1), N[idx].sum(1)
    rate = np.where(bn > 0, be / np.where(bn > 0, bn, 1), np.nan)
    lo, hi = np.nanpercentile(rate, 2.5, axis=0), np.nanpercentile(rate, 97.5, axis=0)
    t = pd.DataFrame({"sifat": task, "model": name, "item": items, "n": N.sum(0).astype(int),
                      "galat": E.sum(0) / N.sum(0), "ci95_lo": lo, "ci95_hi": hi})
    t = t.sort_values("galat", ascending=False).reset_index(drop=True)
    t["peringkat"] = np.arange(1, len(t) + 1)
    thr = t.ci95_hi.iloc[5] if len(t) > 5 else np.inf   # peringkat bermakna bila CI terpisah dari peringkat ke-6
    t["ci_terpisah_dari_peringkat_6"] = t.ci95_lo > thr
    return t


# ---------- program utama ----------

def build_data(xgb_run, w2v_run, meta, mask_fn=None, extra_runs=None):
    """Susun {task: (tab, resp_order, strata)}; mask_fn(df, meta)->bool Series memilih rekaman yang dipakai."""
    data = {}
    for task in TASKS:
        xgb = load_preds(xgb_run, task, "test")
        w = load_preds(w2v_run, task, "test")
        w_seeds = {s: g.reset_index(drop=True) for s, g in w.groupby("seed")}
        extra = {}
        for name, (run, fname) in (extra_runs or {}).items():
            e = pd.read_csv(Path(run) / f"{task}_{fname}.csv")
            if "seed" in e.columns:
                extra[name] = {s: g.reset_index(drop=True) for s, g in e.groupby("seed")}
            else:
                extra[name] = {0: e}
        if mask_fn is not None:
            keep = set(mask_fn(xgb, meta))
            xgb = xgb[xgb.source_id.isin(keep)]
            w_seeds = {s: g[g.source_id.isin(keep)] for s, g in w_seeds.items()}
            extra = {n: {s: g[g.source_id.isin(keep)] for s, g in d.items()} for n, d in extra.items()}
        resp_order = sorted(xgb.responden.unique())
        strata = [r.split("-")[0] for r in resp_order]
        data[task] = (model_table(task, xgb, w_seeds, resp_order, extra), resp_order, strata)
    return data


def not_error(df, meta):
    m = meta.loc[df.source_id]
    return df.source_id[~m.flag_error.astype(bool).values]


def dedup_within_folder(df, meta):
    """Pertahankan satu rekaman per isi audio identik (md5) di dalam folder responden."""
    m = meta.loc[df.source_id][["responden", "md5"]].reset_index()
    return m.drop_duplicates(["responden", "md5"], keep="first").source_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xgb", required=True)
    ap.add_argument("--w2v", required=True)
    ap.add_argument("--frozen", help="run encoder beku + probe linear (opsional)")
    ap.add_argument("--xgb-seeds", help="run XGBoost ext dengan *_predictions_test_seeds.csv (opsional)")
    ap.add_argument("--out", default="reports/analysis")
    ap.add_argument("--quick", action="store_true", help="B kecil untuk uji cepat")
    args = ap.parse_args()
    B = 500 if args.quick else B_MAIN
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    rng = np.random.default_rng(RNG_SEED)

    # 1. metrik uji + aturan acuan
    rows = []
    for task in TASKS:
        xgb = load_preds(args.xgb, task, "test")
        w = load_preds(args.w2v, task, "test")
        for cat in ("all", "benar", "salah"):
            sel = lambda d: d if cat == "all" else d[d.category == cat]
            c = counts(sel(xgb).y_true, sel(xgb).y_pred)
            rows.append({"sifat": task, "model": "xgb_seed42", "subset": cat, "n": int(c.sum()),
                         "macro_f1": float(macro_f1_c(c)), "balanced_acc": float(bal_acc_c(c)), "akurasi": float(acc_c(c))})
            vals = [counts(sel(g).y_true, sel(g).y_pred) for _, g in w.groupby("seed")]
            rows.append({"sifat": task, "model": "w2v_bert_rata2_3seed", "subset": cat, "n": int(vals[0].sum()),
                         "macro_f1": float(np.mean([macro_f1_c(v) for v in vals])),
                         "balanced_acc": float(np.mean([bal_acc_c(v) for v in vals])),
                         "akurasi": float(np.mean([acc_c(v) for v in vals]))})
        for rule, pred in reference_rules(task, meta, xgb).items():
            for cat in ("all", "benar", "salah"):
                msk = np.ones(len(xgb), bool) if cat == "all" else (xgb.category == cat).values
                c = counts(xgb.y_true[msk], pred[msk])
                rows.append({"sifat": task, "model": f"acuan_{rule}", "subset": cat, "n": int(c.sum()),
                             "macro_f1": float(macro_f1_c(c)), "balanced_acc": float(bal_acc_c(c)),
                             "akurasi": float(acc_c(c))})
    pd.DataFrame(rows).round(4).to_csv(out / "metrik_uji_dan_acuan.csv", index=False)

    # 2. delta + CI + permutasi + Holm, dan 4. sensitivitas
    extra_runs = {}
    if args.frozen:
        extra_runs["w2v_beku_probe"] = (args.frozen, "predictions_test")
    if args.xgb_seeds:
        extra_runs["xgb_3seed"] = (args.xgb_seeds, "predictions_test_seeds")
    variants = {"utama": None, "tanpa_berkas_error": not_error, "tanpa_salinan_identik": dedup_within_folder}
    res = []
    for label, fn in variants.items():
        data = build_data(args.xgb, args.w2v, meta, fn, extra_runs)
        res.append(analyze_delta(data, label, B))
        if label == "utama":
            if "w2v_beku_probe" in extra_runs:
                res.append(analyze_delta(data, "utama_vs_beku", B, ref="w2v_beku_probe"))
            if "xgb_3seed" in extra_runs:
                res.append(analyze_delta(data, "utama_vs_xgb3seed", B, ref="xgb_3seed"))
            if "w2v_beku_probe" in extra_runs:
                res.append(analyze_delta(data, "beku_vs_xgb", B, model="w2v_beku_probe", ref="xgb"))
    pd.concat(res).round(4).to_csv(out / "delta_ci_permutasi.csv", index=False)

    # 3. LIGO per kategori
    lrows = []
    runs = {"xgb": args.xgb, "w2v_bert": args.w2v}
    if args.frozen:
        runs["w2v_beku_probe"] = args.frozen
    for task in TASKS:
        lrows += ligo_table(task, runs)
    pd.DataFrame(lrows).round(4).to_csv(out / "ligo_per_kategori.csv", index=False)

    # 5. galat per item
    tabs = []
    for task in TASKS:
        tabs.append(item_errors(task, "xgb", load_preds(args.xgb, task, "test"), rng))
        w = load_preds(args.w2v, task, "test")
        tabs.append(item_errors(task, "w2v_bert", w[w.seed == w.seed.min()], rng))
    pd.concat(tabs).round(4).to_csv(out / "galat_per_item.csv", index=False)
    print("selesai ->", out)


if __name__ == "__main__":
    main()
