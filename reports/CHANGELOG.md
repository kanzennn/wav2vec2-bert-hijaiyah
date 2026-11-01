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

## 2026-11-02 — Deep learning: CNN dan CRNN (log-Mel)

- Fitur log-Mel 64 pita (25/10 ms, 64 x 191 untuk 1,9 s), normalisasi per pita dari frame valid data latih tiap run,
  frame padding = 0 dan diabaikan saat pooling (CNN: rata-rata waktu bermasker; CRNN: GRU di-pack). Commit ea8df55.
- Protokol per seed (42, 43, 44): CV fold 1-5 dengan early stopping (macro-F1, patience 8, maks 40 epoch);
  epoch akhir = median epoch terbaik CV; latih ulang di seluruh latih, uji sekali; LIGO dengan epoch yang sama.
  Hiperparameter belum di-tuning; tanpa augmentasi.
- Run `20261102-011321_cnn_mel_all` dan `20261102-025730_crnn_mel_all`. Macro-F1 uji (mean +- std 3 seed):
  CNN S1 0,665 +- 0,012, S3 0,605 +- 0,019, T2 0,705 +- 0,022; CRNN S1 0,726 +- 0,015, S3 0,722 +- 0,013, T2 0,744 +- 0,024.
  LIGO pooled: CNN 0,570 / 0,592 / 0,640; CRNN 0,669 / 0,629 / 0,651.
- CRNN terbaik di ketiga sifat (uji dan LIGO). CNN tidak lebih baik dari model klasik kecuali T2, dan S3-nya turun jauh dari CV (0,679) ke uji (0,605).
- Variasi antar-fold CV besar untuk deep learning (mis. CRNN S1: fold 1 0,596 sampai fold 3 0,858).
