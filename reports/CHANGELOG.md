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
