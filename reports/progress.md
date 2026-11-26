# Progress eksperimen lanjutan (CLAUDE.md v7 bagian 10)

Diperbarui tiap selesai satu langkah supaya sesi Claude Code berikutnya (di laptop mana pun) bisa melanjutkan tanpa menebak.

| Tanggal | Eksperimen | Status | Run id / berkas | Catatan |
|---|---|---|---|---|
| 2026-11-24 | E1 XGBoost grid diperluas + 3 seed | selesai (laptop Faiz) | `runs/20261123-234403_xgb_mfcc_ext_all` | sel terpilih masih di tepi grid (max_depth 8; T2 n_estimators 1000), dicatat sebagai keterbatasan |
| 2026-11-26 | E2 sensitivitas tanpa 26 berkas `-error` | **siap dijalankan di laptop teman** | config `configs/xgb_mfcc_ext_noerror.yaml` | kode + tes di commit c568487; smoke lolos di laptop Faiz. Run penuh dihentikan atas permintaan Faiz (`*_aborted`, tidak dipakai). Perintah: `python -m ta.train --config configs/xgb_mfcc_ext_noerror.yaml --smoke`, lalu `--task all`. Bandingkan macro-F1 uji dengan E1 (3 seed) |
| 2026-11-23 s/d 26 | E3 encoder beku + probe linear | berjalan; dipindah ke laptop teman | `runs/20261123-190801_w2v_bert_frozen_all` | selesai: CV S1 fold 1-3 (laptop Faiz, RTX 4060 8 GB). CV S1 fold 4 diulang dari epoch 1. Lanjutkan dengan `--resume` (notebook `notebooks/lanjut_training_laptop_lain.ipynb`) |
| 2026-11-26 | E5 galat per huruf induk (28 huruf) | selesai (laptop Faiz) | `reports/analysis/galat_per_item.csv` | commit 2e4d59f; tabel 36 item lama di `reports/analysis/archive_36item/`. Tidak ada huruf yang CI-nya terpisah dari peringkat ke-6 |
| - | E0 pemeriksaan laptop teman | belum | `reports/hardware_laptop.md` | wajib sebelum E3/E4 |
| - | E4 biaya komputasi | belum | `src/ta/cost.py` belum ada | diukur di laptop teman untuk kedua model |
| - | E6 analisis statistik akhir | belum | - | setelah E3 selesai; pakai `--frozen` dan `--xgb-seeds runs/20261123-234403_xgb_mfcc_ext_all` |
| - | E7 baseline durasi | opsional, belum disetujui | - | - |
