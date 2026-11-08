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

## 2026-11-02 — Jeda/lanjut untuk training deep learning

- `python -m ta.train --resume <run_dir>`: tiap unit (sifat, seed, cv/test/ligo, fold) disimpan ke `run_dir/units/` begitu selesai,
  model yang sedang dilatih di-checkpoint tiap epoch ke `run_dir/checkpoints/` (model, optimizer, scaler AMP, bobot terbaik, RNG).
  Resume memakai `config.yaml` dari run_dir dan menolak config/--task yang berbeda. Baris `reports/results.csv` dengan run_id yang sama diganti, tidak dobel.
- Diuji: smoke CNN yang dibunuh dua kali di tengah lalu dilanjutkan menghasilkan 10 berkas csv identik dengan run tanpa jeda.
- Run fine-tune wav2vec `20261102-083223_w2v_finetune_all` dimulai dengan kode sebelum fitur ini, jadi run itu belum bisa dilanjutkan bila dihentikan.

## 2026-11-09 — CLAUDE.md versi 5: XGBoost vs Wav2Vec2-BERT

- CLAUDE.md diganti ke versi 5 (judul "Peningkatan Klasifikasi Sifat Huruf Hijaiyah Menggunakan Wav2Vec2-BERT"; hanya dua model).
- **Keputusan user (2026-11-09):** SVM, RF, CNN, CRNN, wav2vec2-base beku, dan fine-tune wav2vec2-base (tidak selesai, mati setelah CV S1 seed 42)
  diarsipkan ke `runs/archive_out_of_scope/` dan `reports/archive_out_of_scope/`; kodenya dihapus dari `src/` (tetap di riwayat git).
  `data/raw_extract/` dihapus untuk menghemat disk. Outlier durasi tetap tidak dibuang.
- Baseline XGBoost (`20261102-004931_xgb_mfcc_all`) tetap dipakai: dilatih di split bawaan, grid dipilih lewat CV fold 1-5.
- Struktur kode: `models/xgboost_model.py`, `models/wav2vec_bert.py`, `features/raw.py` (fbank dari `SeamlessM4TFeatureExtractor` bawaan checkpoint).
- Checkpoint `facebook/w2v-bert-2.0` revisi `da985ba0987f70aaeb84a80f2851cfac8c697a7b` (model.safetensors) di `pretrained/` (tidak di-commit).
  Ditambah `transformers==4.51.3` dan `bitsandbytes==0.45.5`.
- Fine-tune seluruh lapisan agar muat di GPU 8 GB: bf16, gradient checkpointing, batch 8 x akumulasi 4, AdamW 8-bit, kepala linear dua kelas.
  Uji coba langkah tunggal: puncak memori GPU 6,2 GB, sekitar 0,6 s per langkah batch 8 (GPU tidak dipakai aplikasi lain).
- Masalah di smoke test: crash (access violation) saat menyalin bobot 2,3 GB ke RAM ketika RAM bebas hanya sekitar 3,5 GB, lalu CUDA out of memory
  karena aplikasi lain memakai sekitar 3,4 GB memori GPU. Perbaikan kode: bobot terbaik ditulis ke disk dan dimuat langsung ke GPU.
  Smoke test penuh belum lolos; perlu GPU dan RAM yang tidak dipakai aplikasi lain.
- Audit: cek responden (40 responden menjadi 37 `grup_cv`, tidak ada responden di dua fold/split) ditambahkan ke `reports/data_audit.md` dan tes anti-leakage.
- `python -m ta.evaluate --improvement` menulis `reports/improvement.csv` (macro-F1 dan akurasi Wav2Vec2-BERT dikurangi XGBoost per sifat, per fold).
