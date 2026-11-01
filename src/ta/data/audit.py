"""Data audit: python -m ta.data.audit

Menulis reports/data_audit.md dan reports/outlier_durasi.csv. Tidak membuang rekaman apa pun.
Sekaligus membangun cache praproses (data/interim/trim16k_*).
"""
import argparse
import math

import numpy as np
import pandas as pd

from ta.data.index import build_index, disk_vs_metadata, load_leakage_exclusions, load_metadata
from ta.data.labels import TASKS
from ta.data.preprocess import DEFAULT_PREPROCESS, build_cache
from ta.data.split import add_ligo_fold, write_splits
from ta.paths import REPORTS_DIR

SHORT_S, LONG_S = 0.3, 2.0


def md_table(df: pd.DataFrame, index=True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def run(n_jobs=4):
    R = ["# Data audit (TAHAP 2)", "", "Dibuat otomatis oleh `python -m ta.data.audit`. Tidak ada rekaman yang dibuang oleh audit ini.", ""]
    meta = load_metadata()
    idx = build_index()
    excl = load_leakage_exclusions()

    # ---- 1. jumlah
    dv = disk_vs_metadata()
    R += ["## 1. Jumlah berkas", "",
          f"- WAV di disk: {dv['n_disk']} ({dv['n_disk_per_category']})",
          f"- Baris metadata: {len(meta)} ({meta.category.value_counts().to_dict()})",
          f"- Di metadata tapi tidak ada di disk: {len(dv['missing_on_disk'])}",
          f"- Di disk tapi tidak ada di metadata: {len(dv['not_in_metadata'])}; sama dengan `rekaman_dikeluarkan.csv`: {dv['not_in_metadata_equals_dataset_excluded']}",
          f"- Dikeluarkan karena kebocoran latih-uji (`exclude_leakage.csv`, keputusan user 2026-11-01): {len(excl)}",
          f"- **Rekaman terpakai: {len(idx)}** (latih {int((idx.split == 'latih').sum())}, uji {int((idx.split == 'uji').sum())})", ""]
    ok_counts = len(meta) == 21591 and len(idx) == 21591 - len(excl) and not dv["missing_on_disk"]

    # ---- 2. kualitas audio (dari cache praproses)
    cdir = build_cache(idx, DEFAULT_PREPROCESS, n_jobs=n_jobs)
    st = pd.read_csv(cdir / "stats.csv")
    a = idx.merge(st, on="source_id", how="left", validate="one_to_one")
    md5_meta_ok = int((a.md5_samples == a.md5).sum())
    R += ["## 2. Kualitas audio", "",
          f"Cache praproses: `{cdir.relative_to(cdir.parents[2]).as_posix()}`, parameter: `{DEFAULT_PREPROCESS}`.", "",
          f"- Berkas gagal dibaca: {int(a.md5_samples.isna().sum())}",
          f"- Sample rate: {a.sr_read.value_counts().to_dict()}; cocok dengan metadata: {bool((a.sr_read == a.sr).all())}",
          f"- Durasi baca ulang vs metadata, selisih maks: {float((a.dur_raw - a.duration).abs().max()):.2e} s",
          f"- md5 sampel (float32) == kolom `md5` metadata: {md5_meta_ok} / {len(a)}",
          f"- Berkas kosong (0 sampel): {int((a.n_raw == 0).sum())}",
          f"- Hampir senyap (RMS < -60 dBFS): {int((a.rms_dbfs_raw < -60).sum())}",
          f"- Ada sampel terpotong/clipping (|x| >= 0,999): {int((a.clip_frac > 0).sum())} berkas; >1% sampel: {int((a.clip_frac > 0.01).sum())}",
          f"- RMS dBFS: median {a.rms_dbfs_raw.median():.1f}, p1 {a.rms_dbfs_raw.quantile(.01):.1f}, p99 {a.rms_dbfs_raw.quantile(.99):.1f}",
          ""]
    quiet = a[a.rms_dbfs_raw < -60][["source_id", "rms_dbfs_raw", "dur_raw"]]
    if len(quiet):
        R += [md_table(quiet.round(3), index=False), ""]

    # ---- 3. durasi + panjang target
    def q(s):
        return {"median": s.median(), "p1": s.quantile(.01), "p95": s.quantile(.95), "p99": s.quantile(.99), "min": s.min(), "max": s.max()}
    dur = pd.DataFrame({"mentah (semua)": q(a.dur_raw), "setelah trim (semua)": q(a.dur_trim),
                        "setelah trim (latih)": q(a[a.split == "latih"].dur_trim)}).T.round(3)
    p99_train = float(a[a.split == "latih"].dur_trim.quantile(.99))
    target_s = math.ceil(p99_train * 10) / 10
    n_crop = int((a.dur_trim > target_s).sum())
    R += ["## 3. Durasi dan panjang tetap", "", md_table(dur), "",
          f"- Mentah > {LONG_S} s: {int((a.dur_raw > LONG_S).sum())}; < {SHORT_S} s: {int((a.dur_raw < SHORT_S).sum())}",
          f"- Setelah trim > {LONG_S} s: {int((a.dur_trim > LONG_S).sum())}; < {SHORT_S} s: {int((a.dur_trim < SHORT_S).sum())}",
          f"- **Persentil 99 durasi setelah trim (data latih) = {p99_train:.3f} s -> panjang tetap = {target_s:.1f} s "
          f"({int(round(target_s * DEFAULT_PREPROCESS['sr']))} sampel)**, dibulatkan ke atas ke 0,1 s.",
          f"- Rekaman yang terpotong (tengah) pada panjang ini: {n_crop} ({n_crop / len(a):.2%}); sisanya di-pad nol di kanan.", ""]

    out = a[(a.dur_raw < SHORT_S) | (a.dur_raw > LONG_S)].copy()
    out["jenis"] = np.where(out.dur_raw < SHORT_S, "pendek(<0.3s)", "panjang(>2s)")
    out_cols = ["source_id", "jenis", "category", "responden", "split", "fold", "item", "sub", "rep", "dur_raw", "dur_trim", "rms_dbfs_raw", "y_S1", "y_S3", "y_T2"]
    out[out_cols].sort_values(["jenis", "dur_raw"]).round(4).to_csv(REPORTS_DIR / "outlier_durasi.csv", index=False)
    dur_cat = a.groupby("category").dur_trim.describe(percentiles=[.01, .5, .99])[["count", "1%", "50%", "99%", "max"]].round(3)
    R += [f"Outlier durasi mentah ditulis ke `reports/outlier_durasi.csv` ({len(out)} baris). **Belum ada yang dibuang**; menunggu keputusan user.", "",
          md_table(out.groupby(["jenis", "category", "split"]).size().rename("n").to_frame()), "",
          "Durasi setelah trim per kategori:", "", md_table(dur_cat), "",
          "**Risiko jalan pintas:** rekaman sangat pendek dan sangat panjang didominasi Data Salah. Karena label ditentukan oleh "
          "(item, harakat, benar/salah), durasi bisa membantu model menebak status benar/salah tanpa belajar sifat artikulasi. "
          "Fitur MFCC (mean + std) dihitung hanya pada frame valid (tanpa padding), jadi durasi tidak masuk langsung sebagai fitur, "
          "tetapi skor per subset Data Benar/Data Salah tetap dilaporkan untuk memantau efek ini.", "",
          "Sebaran outlier pendek per responden (10 terbanyak):", "",
          md_table(out[out.jenis.str.startswith("pendek")].groupby("responden").size().nlargest(10).rename("n").to_frame()), ""]

    # ---- 4. kebocoran isi
    R += ["## 4. Pemeriksaan kebocoran isi (md5)", ""]
    for col, name in [("md5_samples", "md5 sampel"), ("md5_file", "md5 byte berkas")]:
        lat, uji = set(a[a.split == "latih"][col]), set(a[a.split == "uji"][col])
        fm = a.groupby(col).fold.nunique()
        R.append(f"- {name}: unik {a[col].nunique()} dari {len(a)}; sama latih-uji: **{len(lat & uji)}**; lintas fold: {int((fm > 1).sum())}")
    cross = a[a.md5_samples.isin(a.groupby("md5_samples").fold.nunique().loc[lambda s: s > 1].index)]
    pairs = cross.groupby("md5_samples").grup_cv.apply(lambda s: " <-> ".join(sorted(set(s)))).value_counts().rename("n_md5").to_frame()
    R += ["", "md5 lintas fold di data latih (dibiarkan, dicatat sebagai keterbatasan CV):", "", md_table(pairs), "",
          "Duplikat isi di dalam satu `grup_cv` (pengulangan identik) tidak memengaruhi split; tidak ada duplikat yang berbeda item, harakat, atau label.", ""]

    # ---- 5. label dan split
    lab = []
    for t in TASKS:
        for name, sub in [("latih", a[a.split == "latih"]), ("uji", a[a.split == "uji"]), ("semua", a)]:
            vc = sub[f"y_{t}"].value_counts()
            lab.append({"sifat": t, "bagian": name, "kelas0": int(vc.get(0, 0)), "kelas1": int(vc.get(1, 0)),
                        "porsi1": round(vc.get(1, 0) / len(sub), 4)})
    lab = pd.DataFrame(lab)
    R += ["## 5. Sebaran label dan split", "", md_table(lab, index=False), "",
          "Data latih seimbang (porsi kelas 1 sekitar 0,5) untuk ketiga sifat, jadi **class weight tidak dipakai**.", "",
          md_table(pd.crosstab(a.split, a.fold)), "",
          f"- `grup_cv` di lebih dari satu fold: {int((a.groupby('grup_cv').fold.nunique() > 1).sum())}; di lebih dari satu split: {int((a.groupby('grup_cv').split.nunique() > 1).sum())}", ""]
    per_fold = a[a.split == "latih"].groupby("fold")[[f"y_{t}" for t in TASKS]].mean().round(3)
    R += ["Porsi kelas 1 per fold latih:", "", md_table(per_fold), ""]

    # ---- 6. LIGO
    write_splits()
    li = add_ligo_fold(a)
    lg = li.groupby("ligo_fold").agg(items=("item", lambda s: ", ".join(sorted(s.unique(), key=lambda x: (int(''.join(c for c in x if c.isdigit())), x)))),
                                     rekaman_latih_item_ini=("split", lambda s: int((s == "latih").sum())),
                                     rekaman_uji_item_ini=("split", lambda s: int((s == "uji").sum())))
    for t in TASKS:
        lg[f"huruf {t}=1 (benar)"] = li[li.category == "benar"].groupby("ligo_fold").apply(
            lambda g: int(g.loc[g[f"y_{t}"] == 1, "huruf_no"].nunique()), include_groups=False)
    R += ["## 6. Lipatan leave-item-group-out", "",
          "Disimpan di `data/splits/ligo_items.csv` dan `data/splits/ligo_folds.csv`. Varian ikut huruf dasarnya. "
          "Evaluasi: latih = responden latih dengan item di luar lipatan k; uji = responden uji dengan item di lipatan k.", "",
          md_table(lg), "",
          "Lipatan 6 tidak berisi huruf qalqalah atau isti'la (Data Benar), karena hanya ada 5 huruf qalqalah untuk 6 lipatan. "
          "Kelas 1 di lipatan itu tetap ada lewat Data Salah (label dibalik).", ""]

    # ---- 7. asumsi
    R += ["## 7. Asumsi", "",
          "- Responden `benar-N` dan `salah-N` diperlakukan sebagai identitas berbeda, mengikuti split bawaan dataset (belum dipastikan apakah orangnya sama).",
          "- Arti akhiran varian (`i`, `ie`, `ik`, `t`) belum diketahui; varian diperlakukan sebagai item biasa.",
          "- 26 berkas `*-error` di `Data Benar/15` diperlakukan sebagai pengucapan benar (sesuai dataset).", ""]

    (REPORTS_DIR / "data_audit.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))
    print(f"\nOK jumlah: {ok_counts}; target panjang: {target_s} s")
    return {"ok_counts": ok_counts, "target_s": target_s, "p99_train": p99_train}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    run(ap.parse_args().n_jobs)
