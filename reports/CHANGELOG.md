# CHANGELOG

## 2026-11-01 — TAHAP 1 (ekstraksi dan inventarisasi)

- Dataset diekstrak dari `Hijaiyah Letter Pronunciation Audio Dataset.zip` ke `data/raw/` (21.600 WAV) dan `data/labels/`.
  Hasil verifikasi ada di `reports/label_inventory.md`.
- Tidak ditemukan hasil lama konsep identifikasi huruf 36 kelas. Folder `reports/archive_letter_id/` dan `runs/archive_letter_id/` hanya berisi README.
- Ditemukan 1 rekaman identik (md5 `ce66fdfe...`, item `21t`, kasrah) yang ada di latih (`benar-11`) dan uji (`benar-12`, `benar-13`).
  **Keputusan user:** 5 salinan di data latih dikeluarkan lewat `data/labels/exclude_leakage.csv`. Data uji tidak diubah.
  Total rekaman terpakai: 21.586 (latih 17.275, uji 4.311).
- 18 md5 lintas fold di dalam data latih dibiarkan dan dicatat sebagai keterbatasan validasi silang.

## 2026-11-01 — TAHAP 2 (audit, label, anti-leakage, baseline MFCC + SVM)

- Paket `src/ta` (index, get_label, praproses, split + LIGO, audit, MFCC, SVM, train, evaluate) dan 21 unit test (label + anti-leakage), semuanya lolos.
- Audit: `reports/data_audit.md`. Panjang tetap 1,9 s (p99 durasi setelah trim di data latih = 1,870 s). 864 outlier durasi
  dicatat di `reports/outlier_durasi.csv`, belum ada yang dibuang. Temuan: durasi berbeda jauh antara Data Benar
  (median 0,95 s) dan Data Salah (0,61 s), sehingga ada risiko jalan pintas.
- Lipatan LIGO: 6 lipatan per huruf (varian ikut huruf dasar), latih = responden latih dengan item lain, uji = responden uji dengan item lipatan.
- Baseline `svm_mfcc` (run `20261101-225111_svm_mfcc_all`, commit 0a70379, tanpa augmentasi). Macro-F1 uji: S1 0,685, S3 0,665, T2 0,666;
  LIGO: 0,609 / 0,610 / 0,617. Baseline mayoritas sekitar 0,333.

## 2026-11-02 — Model klasik lanjutan (Random Forest, XGBoost)

- `make_model` mendukung `rf` dan `xgb`; config bisa punya `model.fixed` (parameter tetap) selain `model.grid`. Commit e708014.
- Run `20261101-234341_rf_mfcc_all` dan `20261102-004931_xgb_mfcc_all`, protokol sama dengan SVM (grid di CV fold 1-5, uji sekali, LIGO tanpa tuning ulang), tanpa augmentasi.
- Macro-F1 uji (RF / XGB / SVM): S1 0,695 / 0,689 / 0,685; S3 0,672 / 0,668 / 0,665; T2 0,674 / 0,662 / 0,666.
  LIGO pooled (RF / XGB / SVM): S1 0,629 / 0,624 / 0,609; S3 0,620 / 0,617 / 0,610; T2 0,626 / 0,611 / 0,617.
- Ketiga model klasik berbeda tipis (selisih uji di bawah 0,02, masih di dalam std antar-fold CV 0,02-0,06). Skor Data Salah tetap sekitar 0,52-0,58 untuk semua model.
- Parameter terbaik RF (max_depth None, min_samples_leaf 1) dan XGB (n_estimators 600, max_depth 6) ada di tepi grid, sama seperti SVM.
